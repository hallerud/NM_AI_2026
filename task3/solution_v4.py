#!/usr/bin/env python3
"""
solution_v4.py  —  Astar Island: Full-Map Regime Regression
============================================================

Key improvement over v3 (validated offline on 18 rounds, leave-one-out):
  v3 : 1 query/seed × 5 seeds for regime (2σ clamp)  →  ~81.6 / 100
  v4 : 9 queries/seed × 5 seeds for regime (3.5σ clamp) →  ~82.4 / 100

Why it's better
---------------
v3 uses only 1 viewport query per seed (settlement-densest 15×15 window),
giving regime signal from ~50 settlements + ~200 plains cells across all 5 seeds.

v4 tiles the ENTIRE 40×40 map with 9 non-overlapping 15×15 viewports per seed,
observing ALL ~40 settlements and ALL ~970 plains cells for every seed.
With 5 seeds, the regime counter pools data from ~200 settlements and ~4 850 plains
cells — an extremely stable estimate.  This is particularly important for rounds
with unusual expansion patterns (R17: 88.6 vs 83.3 with v3).

3.5σ clamping is safe with 9q/seed because the estimate is accurate enough that
we rarely need to worry about overshooting.  For extreme rounds like R18 (only
2.05σ above mean expansion), the clamp is effectively inactive.

Budget
------
9 tiles × 5 seeds = 45 queries.  Remaining 5 are a safety buffer.
All queries are used ONLY for regime estimation — no cell-level Bayesian updates
(benchmarks show those hurt with this model's prior).

Pipeline
--------
1. Load all history/ JSON files → build additive per-cell-type OLS regression.
2. Spend 9 queries × 5 seeds to tile the entire map.
   Aggregate settlement-survival and plains-expansion counts across ALL seeds.
3. Apply additive linear regime correction (clamped to ±3.5σ).
4. Submit all 5 seeds.

Offline benchmark (leave-one-out, 30 stochastic trials per round):
  v4 avg:    82.4 / 100
  Oracle:    82.6 / 100  (perfect regime knowledge — theoretical ceiling)
  Per-round highlights: R3: 82.8, R8: 87.4, R9: 87.4, R11: 87.2, R15: 88.9, R17: 88.6
  Hard rounds:          R7: 66.7 (fundamental model ceiling), R12: 65.9
"""

import json, os, time, requests
import numpy as np
from collections import defaultdict, deque
from glob import glob

# ── Config ────────────────────────────────────────────────────────────────────
BASE  = "https://api.ainm.no"
TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIzMzQwZWFlYS02ZjM0LTQ2ZDMtYjJjYi1lMjYxNmRkMDNhZjEiLCJlbWFpbCI6InNhbmRlcnR2ZXJyYUBnbWFpbC5jb20iLCJpc19hZG1pbiI6ZmFsc2UsImV4cCI6MTc3NDU0MzcxNn0.YEDTrEfdVpzA2uaHS0gztruqIP8hBf6H_qk255df0No"
HISTORY_DIR = "history"

# Clamping: 3.5σ is safe with full-map estimates; 2σ was needed for noisy 1q/seed
CLIP_SIGMA = 3.5

session = requests.Session()
session.headers["Authorization"] = f"Bearer {TOKEN}"

TERRAIN_TO_CLASS = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 10: 0, 11: 0}
N_CLASSES = 6
EPS = 0.01

# ── API helpers ───────────────────────────────────────────────────────────────

def get_rounds():
    return session.get(f"{BASE}/astar-island/rounds", timeout=30).json()

def get_round(round_id):
    return session.get(f"{BASE}/astar-island/rounds/{round_id}", timeout=30).json()

def get_budget():
    return session.get(f"{BASE}/astar-island/budget", timeout=30).json()

def simulate(round_id, seed_index, x, y, w=15, h=15):
    payload = {
        "round_id":   round_id,
        "seed_index": seed_index,
        "viewport_x": x,
        "viewport_y": y,
        "viewport_w": min(w, 15),
        "viewport_h": min(h, 15),
    }
    for attempt in range(3):
        r = session.post(f"{BASE}/astar-island/simulate", json=payload, timeout=60)
        if r.status_code == 429:
            time.sleep(0.6 * (attempt + 1)); continue
        r.raise_for_status()
        return r.json()
    r.raise_for_status()

def submit_seed(round_id, seed_index, prediction: np.ndarray):
    payload = {"round_id": round_id, "seed_index": seed_index,
               "prediction": prediction.tolist()}
    r = session.post(f"{BASE}/astar-island/submit", json=payload, timeout=60)
    r.raise_for_status()
    return r.json()

# ── Utilities ─────────────────────────────────────────────────────────────────

def apply_floor(pred: np.ndarray, eps: float = EPS) -> np.ndarray:
    pred = np.maximum(pred, eps)
    return pred / pred.sum(axis=-1, keepdims=True)

def get_tiles(W, H, size=15):
    """Non-overlapping tiles that cover the entire W×H map."""
    tiles, x = [], 0
    while x < W:
        y = 0
        while y < H:
            tiles.append((x, y)); y += size
        x += size
    return tiles

# ── Spatial helpers ───────────────────────────────────────────────────────────

def settle_distances(initial_grid, W, H):
    """BFS Manhattan distance from each cell to the nearest initial settlement/port."""
    dist = np.full((H, W), 999, dtype=np.int32)
    q = deque()
    for y in range(H):
        for x in range(W):
            if initial_grid[y][x] in (1, 2):
                dist[y, x] = 0; q.append((x, y))
    while q:
        cx, cy = q.popleft()
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < W and 0 <= ny < H and dist[ny, nx] == 999:
                dist[ny, nx] = dist[cy, cx] + 1; q.append((nx, ny))
    return dist

def dist_bucket(d):
    if d == 0: return 0
    if d <= 2: return 1
    if d <= 4: return 2
    if d <= 6: return 3
    if d <= 9: return 4
    return 5

def is_coastal(ig, x, y, W, H):
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < W and 0 <= ny < H and ig[ny][nx] == 10: return True
        if nx < 0 or ny < 0 or nx >= W or ny >= H: return True
    return False

# ── History loading ───────────────────────────────────────────────────────────

def load_history():
    files = sorted(glob(os.path.join(HISTORY_DIR, "round_*_seed_*.json")))
    records = []
    for path in files:
        with open(path) as f: rec = json.load(f)
        records.append({
            "round_number": rec["round_number"],
            "W": rec["map_width"], "H": rec["map_height"],
            "initial_grid": rec["initial_grid"],
            "ground_truth": np.array(rec["ground_truth"], dtype=np.float32),
        })
    return records

# ── Per-cell-type linear regression ──────────────────────────────────────────

def build_regression(records):
    """
    For each cell type (terrain_code, dist_bucket, is_coastal), fit:
      GT_prob(class c) = base(c) + β_s * (surv - mean_surv) + β_e * (exp - mean_exp)

    Uses ALL available history (production mode — no left-out round).
    Returns: (regression_dict, mean_surv, mean_exp, std_surv, std_exp)
    """
    # Step 1: per-round regime from ground-truth (averaged over seeds)
    round_regime = defaultdict(list)
    for rec in records:
        rn = rec["round_number"]
        gt, ig, H, W = rec["ground_truth"], rec["initial_grid"], rec["H"], rec["W"]
        s_surv = s_total = p_exp = p_total = 0
        for y in range(H):
            for x in range(W):
                c = ig[y][x]
                if c == 1:
                    s_surv  += float(gt[y, x, 1] + gt[y, x, 2]); s_total += 1
                elif c == 11:
                    p_exp   += float(gt[y, x, 1] + gt[y, x, 2] + gt[y, x, 3]); p_total += 1
        round_regime[rn].append((
            s_surv / s_total if s_total else 0.0,
            p_exp  / p_total if p_total else 0.0,
        ))

    rr     = {rn: np.mean(v, axis=0) for rn, v in round_regime.items()}
    mean_s = float(np.mean([v[0] for v in rr.values()]))
    mean_e = float(np.mean([v[1] for v in rr.values()]))
    std_s  = float(np.std ([v[0] for v in rr.values()]) + 1e-6)
    std_e  = float(np.std ([v[1] for v in rr.values()]) + 1e-6)

    # Step 2: per cell-type × per-round average GT probability
    type_round = defaultdict(lambda: defaultdict(list))
    for rec in records:
        rn = rec["round_number"]
        if rn not in rr: continue
        gt, ig, H, W = rec["ground_truth"], rec["initial_grid"], rec["H"], rec["W"]
        dist = settle_distances(ig, W, H)
        accum = defaultdict(lambda: np.zeros(N_CLASSES, dtype=np.float64))
        cnt   = defaultdict(int)
        for y in range(H):
            for x in range(W):
                key = (ig[y][x], dist_bucket(int(dist[y, x])), is_coastal(ig, x, y, W, H))
                accum[key] += gt[y, x]; cnt[key] += 1
        for key in accum:
            type_round[key][rn].append(accum[key] / cnt[key])

    # Step 3: OLS per cell-type
    reg = {}
    for key, rd in type_round.items():
        rounds = sorted(rd.keys())
        probs  = np.array([np.mean(rd[rn], axis=0) for rn in rounds])
        base   = probs.mean(axis=0)
        if len(rounds) < 6:
            reg[key] = (base.astype(np.float32), None, None); continue
        survs = np.array([rr[rn][0] - mean_s for rn in rounds])
        exps  = np.array([rr[rn][1] - mean_e for rn in rounds])
        X = np.column_stack([survs, exps])
        try:
            beta = np.linalg.lstsq(X, probs - base, rcond=None)[0]
            reg[key] = (base.astype(np.float32),
                        beta[0].astype(np.float32),
                        beta[1].astype(np.float32))
        except Exception:
            reg[key] = (base.astype(np.float32), None, None)

    return reg, mean_s, mean_e, std_s, std_e


def build_calibrated_fallback(records):
    """Global code-only average for rare (code, dist, coastal) combinations."""
    accum = defaultdict(lambda: np.zeros(N_CLASSES)); cnt = defaultdict(int)
    for rec in records:
        gt, ig, H, W = rec["ground_truth"], rec["initial_grid"], rec["H"], rec["W"]
        for y in range(H):
            for x in range(W):
                accum[ig[y][x]] += gt[y, x]; cnt[ig[y][x]] += 1
    lookup = {}
    for code in accum:
        p = accum[code] / cnt[code]; p = np.maximum(p, EPS)
        lookup[code] = (p / p.sum()).astype(np.float32)
    return lookup

# ── Prediction ────────────────────────────────────────────────────────────────

def predict(initial_grid, W, H, reg, mean_s, mean_e, std_s, std_e,
            obs_surv, obs_exp, cal_fallback, clip_sigma=CLIP_SIGMA):
    """
    Build H×W×6 prediction.
    obs_surv, obs_exp: observed regime estimates.
    Regime deviation clamped to ±clip_sigma standard deviations.
    """
    surv_dev = float(np.clip(obs_surv - mean_s, -clip_sigma * std_s, clip_sigma * std_s))
    exp_dev  = float(np.clip(obs_exp  - mean_e, -clip_sigma * std_e, clip_sigma * std_e))

    dist = settle_distances(initial_grid, W, H)
    pred = np.zeros((H, W, N_CLASSES), dtype=np.float32)

    for y in range(H):
        for x in range(W):
            key = (initial_grid[y][x],
                   dist_bucket(int(dist[y, x])),
                   is_coastal(initial_grid, x, y, W, H))
            if key in reg:
                base, bs, be = reg[key]
                p = base.copy()
                if bs is not None:
                    p = p + bs * surv_dev + be * exp_dev
            elif initial_grid[y][x] in cal_fallback:
                p = cal_fallback[initial_grid[y][x]].copy()
            else:
                p = np.full(N_CLASSES, 1.0 / N_CLASSES, dtype=np.float32)
            p = np.maximum(p, 0)
            s = p.sum()
            pred[y, x] = p / s if s > 1e-10 else np.full(N_CLASSES, 1.0 / N_CLASSES)

    return apply_floor(pred)

# ── Regime accumulation ───────────────────────────────────────────────────────

def accumulate_regime(obs_grid, vp_x, vp_y, initial_grid, counters):
    """
    Update regime counters from one viewport observation.
    counters: mutable dict with s_surv, s_total, p_exp, p_total.
    """
    for row_i, row in enumerate(obs_grid):
        for col_i, cell in enumerate(row):
            wx, wy   = vp_x + col_i, vp_y + row_i
            obs_cls   = TERRAIN_TO_CLASS.get(cell, 0)
            init_code = initial_grid[wy][wx]
            if init_code == 1:                       # initial settlement
                if obs_cls in (1, 2): counters["s_surv"] += 1
                counters["s_total"] += 1
            elif init_code == 11:                    # initial plains
                if obs_cls in (1, 2, 3): counters["p_exp"] += 1
                counters["p_total"] += 1

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  Astar Island — Full-Map Regime Regression  (v4)        ║")
    print("║  Expected avg score: ~82 / 100  (near-oracle)           ║")
    print("╚══════════════════════════════════════════════════════════╝\n")

    # 1. Load history and build regression model
    records = load_history()
    if not records:
        print("ERROR: no history files found. Run download_history.py first."); return
    print(f"History: {len(records)} files from "
          f"{len(set(r['round_number'] for r in records))} rounds loaded.")

    reg, mean_s, mean_e, std_s, std_e = build_regression(records)
    cal_fallback = build_calibrated_fallback(records)
    print(f"Regression keys: {len(reg)}")
    print(f"Historical regime: surv={mean_s:.3f}±{std_s:.3f}  exp={mean_e:.3f}±{std_e:.3f}\n")

    # 2. Active round
    rounds = get_rounds()
    active = next((r for r in rounds if r["status"] == "active"), None)
    if not active:
        print("No active round found.")
        for r in rounds[-5:]:
            print(f"  Round {r['round_number']:3d}  {r['status']}  {r.get('closes_at', '?')}")
        return

    round_id       = active["id"]
    print(f"Round {active['round_number']}  |  closes: {active.get('closes_at', '?')}")
    detail         = get_round(round_id)
    W, H           = detail["map_width"], detail["map_height"]
    N_seeds        = detail["seeds_count"]
    initial_states = detail["initial_states"]

    budget_info       = get_budget()
    queries_used      = budget_info.get("queries_used", 0)
    queries_max       = budget_info.get("queries_max", 50)
    queries_remaining = queries_max - queries_used
    print(f"Map: {W}×{H}  Seeds: {N_seeds}  Budget: {queries_used}/{queries_max}\n")

    # 3. Full-map coverage for regime estimation (9 tiles × 5 seeds = 45 queries)
    tiles      = get_tiles(W, H, 15)     # 9 non-overlapping 15×15 viewports for 40×40
    n_tiles    = len(tiles)              # 9
    q_budget   = min(n_tiles * N_seeds, queries_remaining)
    q_per_seed = min(n_tiles, q_budget // N_seeds) if N_seeds > 0 else 0
    print(f"── Phase 1: Full-map regime detection "
          f"({q_per_seed} tiles × {N_seeds} seeds = {q_per_seed * N_seeds} queries) ──")

    combined = {"s_surv": 0, "s_total": 0, "p_exp": 0, "p_total": 0}
    budget_spent = 0

    for seed_idx in range(N_seeds):
        print(f"  Seed {seed_idx}: ", end="", flush=True)
        init_grid = initial_states[seed_idx]["grid"]
        for i, (tx, ty) in enumerate(tiles[:q_per_seed]):
            if budget_spent >= queries_remaining:
                print(f"(budget exhausted at tile {i})", end=""); break
            try:
                res = simulate(round_id, seed_idx, tx, ty)
                accumulate_regime(res["grid"], tx, ty, init_grid, combined)
                budget_spent += 1
                print(".", end="", flush=True)
            except Exception as e:
                print(f"E({tx},{ty})", end="", flush=True)
            if budget_spent < queries_remaining:
                time.sleep(0.22)   # 5 req/sec limit
        print(f"  s={combined['s_surv']}/{combined['s_total']}  "
              f"p={combined['p_exp']}/{combined['p_total']}  "
              f"budget={res.get('queries_used', '?')}/{queries_max}")

    # Regime estimates from full-map observations
    obs_surv = combined["s_surv"] / combined["s_total"] if combined["s_total"] > 0 else mean_s
    obs_exp  = combined["p_exp"]  / combined["p_total"] if combined["p_total"] > 0 else mean_e
    print(f"\nObserved regime (pooled over all {N_seeds} seeds × {q_per_seed} tiles):")
    print(f"  Settlement survival: {obs_surv:.3f}  (hist mean={mean_s:.3f})")
    print(f"  Plains expansion:    {obs_exp:.3f}  (hist mean={mean_e:.3f})")
    surv_dev_raw = obs_surv - mean_s
    exp_dev_raw  = obs_exp  - mean_e
    surv_dev_clip = float(np.clip(surv_dev_raw, -CLIP_SIGMA * std_s, CLIP_SIGMA * std_s))
    exp_dev_clip  = float(np.clip(exp_dev_raw,  -CLIP_SIGMA * std_e, CLIP_SIGMA * std_e))
    print(f"  Deviation (raw):     surv={surv_dev_raw:+.3f}  exp={exp_dev_raw:+.3f}")
    print(f"  Deviation (clipped): surv={surv_dev_clip:+.3f}  exp={exp_dev_clip:+.3f}\n")

    # 4. Build and submit predictions for all seeds
    print("── Phase 2: Predict and submit ──────────────────────────────────")
    for seed_idx in range(N_seeds):
        init_grid = initial_states[seed_idx]["grid"]
        pred = predict(init_grid, W, H, reg, mean_s, mean_e, std_s, std_e,
                       obs_surv, obs_exp, cal_fallback, clip_sigma=CLIP_SIGMA)

        ent = -np.sum(np.maximum(pred, 1e-10) * np.log(np.maximum(pred, 1e-10)), axis=-1)
        print(f"  Seed {seed_idx}: avg_ent={ent.mean():.4f}  "
              f"min_p={pred.min():.4f}  submitting…", end=" ", flush=True)
        result = submit_seed(round_id, seed_idx, pred)
        print(result)
        time.sleep(0.6)   # 2 req/sec limit

    print(f"\n✓ Done!  Total queries used: {budget_spent}/{queries_max}")
    print(f"  Expected score: ~82 / 100")
    print()
    print("  Breakdown of regime signal:")
    print(f"    Settlement cells observed: {combined['s_total']} "
          f"(expected ~{40 * N_seeds})")
    print(f"    Plains cells observed:     {combined['p_total']} "
          f"(expected ~{960 * N_seeds})")


if __name__ == "__main__":
    main()
