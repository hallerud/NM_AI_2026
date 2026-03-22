#!/usr/bin/env python3
"""
solution_v5.py  —  Astar Island: Regression + DilatedConvNet Ensemble
======================================================================

LOO benchmark results (19-channel features, 18 rounds, 20 trials each):
  reg_orc : 82.62   ← regression with oracle regime
  reg_9q  : 82.35   ← regression with 9q×5s estimated regime
  dil_orc : 80.68   ← DilatedConvNet (31×31 RF) with oracle regime
  dil_k20 : 81.05   ← neural + Bayesian update at κ=20
  ens_orc : 83.12   ← 50/50 ensemble, oracle regime
  ens_k10 : 83.35   ← 50/50 ensemble, Bayesian κ=10

Production config tuned from results:
  ENSEMBLE_W  = 0.72   → 72% regression (reg oracle 1.94 pts better than neural)
  BAYES_KAPPA = 20.0   → k20 beats k10 (81.05 vs 80.82); trend favours higher κ
  Expected live score: ~83.5

How it works
------------
1. Load all history/ JSON files.
2. Train DilatedConvNet on ALL historical data (or load cached weights from
   dilated_weights.pt if the weights are newer than history + unet.py).
3. Spend 9 queries × 5 seeds = 45 queries to tile the entire 40×40 map.
   Pool observations for regime estimation AND per-cell Bayesian update.
4. For each seed: blend regression + Bayesian-updated DilatedConvNet (72/28).
5. Submit all 5 seeds.

GPU
---
Training uses CUDA automatically (via unet.py DEVICE).
RTX 5080: ~30–60 s for 300 epochs on 90 maps × 8 aug = 720 samples.

Budget
------
9 tiles × 5 seeds = 45 queries.  5 remain as safety buffer.
"""

import json, os, sys, time, requests
import numpy as np
from collections import defaultdict, deque
from glob import glob

# ── Import U-Net from unet.py (same directory) ────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from unet import (DilatedConvNet,
                  train_model, predict_map, build_dataset,
                  compute_regime_per_round, compute_regime_stats,
                  bayesian_cell_update,
                  blend_predictions, DEVICE, N_CLASSES, EPS)

# ── Config ────────────────────────────────────────────────────────────────────
BASE        = "https://api.ainm.no"
TOKEN       = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIzMzQwZWFlYS02ZjM0LTQ2ZDMtYjJjYi1lMjYxNmRkMDNhZjEiLCJlbWFpbCI6InNhbmRlcnR2ZXJyYUBnbWFpbC5jb20iLCJpc19hZG1pbiI6ZmFsc2UsImV4cCI6MTc3NDU0MzcxNn0.YEDTrEfdVpzA2uaHS0gztruqIP8hBf6H_qk255df0No"
HISTORY_DIR = "history"
WEIGHTS_DL  = "dilated_weights.pt"  # cached DilatedConvNet weights
REG_STATS   = "regime_stats.json"   # cached regime standardisation stats

# Training hyperparams for the production model
TRAIN_EPOCHS = 300
TRAIN_LR     = 3e-3
TRAIN_BATCH  = 64        # GPU can handle large batches

# ── Tuned from LOO benchmark (19-ch, 18 rounds, 20 trials) ───────────────────
# reg_orc=82.62 > dil_orc=80.68 → tilt ensemble toward regression
ENSEMBLE_W   = 0.72      # regression weight (28% neural); up from 0.5
# k20=81.05 > k10=80.82 > no-bay=80.39 → higher κ is better
BAYES_KAPPA  = 20.0      # Bayesian prior strength; was 10.0

# Regime clamping: 3.5σ is safe with full-map estimates
CLIP_SIGMA = 3.5

TERRAIN_TO_CLASS = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 10: 0, 11: 0}

session = requests.Session()
session.headers["Authorization"] = f"Bearer {TOKEN}"

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
    """Non-overlapping tiles that cover the entire W×H map (9 tiles for 40×40)."""
    tiles, x = [], 0
    while x < W:
        y = 0
        while y < H:
            tiles.append((x, y)); y += size
        x += size
    return tiles

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

# ── Additive regression (for ensemble) ────────────────────────────────────────

def _settle_distances_local(ig, W, H):
    from collections import deque as dq
    dist = np.full((H, W), 999, dtype=np.int32); q = dq()
    for y in range(H):
        for x in range(W):
            if ig[y][x] in (1, 2): dist[y, x] = 0; q.append((x, y))
    while q:
        cx, cy = q.popleft()
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nx, ny = cx+dx, cy+dy
            if 0<=nx<W and 0<=ny<H and dist[ny,nx]==999:
                dist[ny,nx] = dist[cy,cx]+1; q.append((nx,ny))
    return dist

def _is_coastal_local(ig, x, y, W, H):
    for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
        nx, ny = x+dx, y+dy
        if 0<=nx<W and 0<=ny<H and ig[ny][nx]==10: return True
        if nx<0 or ny<0 or nx>=W or ny>=H: return True
    return False

def _dist_bucket(d):
    if d==0: return 0
    if d<=2: return 1
    if d<=4: return 2
    if d<=6: return 3
    if d<=9: return 4
    return 5

def build_regression(records):
    """Build additive OLS regression on ALL records (production mode)."""
    rr_d = defaultdict(list)
    for rec in records:
        rn,gt,ig,H,W = rec["round_number"],rec["ground_truth"],rec["initial_grid"],rec["H"],rec["W"]
        ss=st=pe=pt=0
        for y in range(H):
            for x in range(W):
                c=ig[y][x]
                if c==1:   ss+=float(gt[y,x,1]+gt[y,x,2]); st+=1
                elif c==11: pe+=float(gt[y,x,1]+gt[y,x,2]+gt[y,x,3]); pt+=1
        rr_d[rn].append((ss/st if st else 0, pe/pt if pt else 0))
    rr = {rn: np.mean(v, axis=0) for rn, v in rr_d.items()}
    ms = float(np.mean([v[0] for v in rr.values()]))
    me = float(np.mean([v[1] for v in rr.values()]))
    ss2 = float(np.std([v[0] for v in rr.values()]) + 1e-6)
    se2 = float(np.std([v[1] for v in rr.values()]) + 1e-6)

    type_round = defaultdict(lambda: defaultdict(list))
    for rec in records:
        rn,gt,ig,H,W = rec["round_number"],rec["ground_truth"],rec["initial_grid"],rec["H"],rec["W"]
        dist = _settle_distances_local(ig, W, H)
        acc = defaultdict(lambda: np.zeros(N_CLASSES, dtype=np.float64)); cnt = defaultdict(int)
        for y in range(H):
            for x in range(W):
                key = (ig[y][x], _dist_bucket(int(dist[y,x])), _is_coastal_local(ig,x,y,W,H))
                acc[key] += gt[y,x]; cnt[key] += 1
        for k in acc: type_round[k][rn].append(acc[k] / cnt[k])
    reg = {}
    for key, rd in type_round.items():
        rounds = sorted(rd.keys())
        probs = np.array([np.mean(rd[rn], axis=0) for rn in rounds])
        base = probs.mean(axis=0)
        if len(rounds) < 6: reg[key] = (base.astype(np.float32), None, None); continue
        sv = np.array([rr[rn][0]-ms for rn in rounds])
        ev = np.array([rr[rn][1]-me for rn in rounds])
        X = np.column_stack([sv, ev])
        try:
            beta = np.linalg.lstsq(X, probs-base, rcond=None)[0]
            reg[key] = (base.astype(np.float32), beta[0].astype(np.float32), beta[1].astype(np.float32))
        except: reg[key] = (base.astype(np.float32), None, None)

    # Fallback
    acc2 = defaultdict(lambda: np.zeros(N_CLASSES)); cnt2 = defaultdict(int)
    for rec in records:
        gt,ig,H,W = rec["ground_truth"],rec["initial_grid"],rec["H"],rec["W"]
        for y in range(H):
            for x in range(W): acc2[ig[y][x]]+=gt[y,x]; cnt2[ig[y][x]]+=1
    cal = {}
    for code in acc2:
        p = acc2[code]/cnt2[code]; p = np.maximum(p, EPS); cal[code] = (p/p.sum()).astype(np.float32)
    return reg, ms, me, ss2, se2, cal

def predict_regression(ig, W, H, reg, ms, me, ss, se, obs_s, obs_e, cal, clip=3.5):
    sd = float(np.clip(obs_s-ms, -clip*ss, clip*ss))
    ed = float(np.clip(obs_e-me, -clip*se, clip*se))
    dist = _settle_distances_local(ig, W, H)
    pred = np.zeros((H, W, N_CLASSES), dtype=np.float32)
    for y in range(H):
        for x in range(W):
            key = (ig[y][x], _dist_bucket(int(dist[y,x])), _is_coastal_local(ig,x,y,W,H))
            if key in reg:
                base, bs, be = reg[key]; p = base.copy()
                if bs is not None: p = p + bs*sd + be*ed
            elif ig[y][x] in cal: p = cal[ig[y][x]].copy()
            else: p = np.full(N_CLASSES, 1/N_CLASSES, dtype=np.float32)
            p = np.maximum(p, 0); s = p.sum()
            pred[y,x] = p/s if s>1e-10 else np.full(N_CLASSES, 1/N_CLASSES)
    pred = np.maximum(pred, EPS)
    return pred / pred.sum(axis=-1, keepdims=True)

# ── Model training / loading ──────────────────────────────────────────────────

def _weights_are_current(path, records):
    """
    True if weights file is newer than all history JSON files AND newer than
    unet.py (catches architecture changes like the 12→19 channel upgrade).
    """
    if not os.path.exists(path):
        return False
    wt = os.path.getmtime(path)
    # Check against history files
    for rec_path in glob(os.path.join(HISTORY_DIR, "round_*_seed_*.json")):
        if os.path.getmtime(rec_path) > wt:
            return False
    # Check against unet.py — architecture changes require retraining
    unet_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "unet.py")
    if os.path.exists(unet_path) and os.path.getmtime(unet_path) > wt:
        return False
    return True


def get_dilated_model(records, regime_stats, force_retrain=False):
    """
    Return a trained DilatedConvNet (31×31 RF, 19 input channels).
    Loads from cached weights if up-to-date; otherwise trains and saves.
    Cache is invalidated when any history file OR unet.py is newer than weights.
    """
    label = "DilatedConvNet (31×31 RF, 19-ch)"

    if not force_retrain and _weights_are_current(WEIGHTS_DL, records):
        print(f"  Loading cached {label} from {WEIGHTS_DL}  …")
        m = DilatedConvNet(ch=64).to(DEVICE)
        m.load_state_dict(torch.load(WEIGHTS_DL, map_location=DEVICE,
                                     weights_only=True))
        m.eval()
        print(f"    Loaded ({sum(p.numel() for p in m.parameters()):,} params)")
        return m

    print(f"  Training {label} (device={DEVICE}, epochs={TRAIN_EPOCHS})  …")
    t0 = time.time()
    rpr     = compute_regime_per_round(records)
    dataset = build_dataset(records, rpr, use_augmentation=True,
                            regime_stats=regime_stats, regime_noise_std=0.3)
    m = train_model(dataset, epochs=TRAIN_EPOCHS, lr=TRAIN_LR,
                    batch=TRAIN_BATCH, dropout=0.1,
                    label_smooth=0.05, model_type="dilated",
                    convnet_ch=64, verbose=True)
    torch.save(m.state_dict(), WEIGHTS_DL)
    print(f"    Done in {time.time() - t0:.1f}s → {WEIGHTS_DL}")
    return m

# ── Regime accumulation from API observations ─────────────────────────────────

def accumulate_regime(obs_grid, vp_x, vp_y, initial_grid, counters):
    """
    Update regime counters from one viewport observation.
    counters: mutable dict with s_surv, s_total, p_exp, p_total.
    The API returns a full 15×15 grid even at map edges (out-of-bounds cells
    are padded), so we must bounds-check before accessing initial_grid.
    """
    ig_h = len(initial_grid)
    ig_w = len(initial_grid[0]) if ig_h > 0 else 0
    for row_i, row in enumerate(obs_grid):
        for col_i, cell in enumerate(row):
            wx, wy = vp_x + col_i, vp_y + row_i
            if not (0 <= wy < ig_h and 0 <= wx < ig_w):
                continue                               # skip padding beyond map edge
            obs_cls   = TERRAIN_TO_CLASS.get(cell, 0)
            init_code = initial_grid[wy][wx]
            if init_code == 1:                        # initial settlement
                if obs_cls in (1, 2): counters["s_surv"] += 1
                counters["s_total"] += 1
            elif init_code == 11:                     # initial plains
                if obs_cls in (1, 2, 3): counters["p_exp"] += 1
                counters["p_total"] += 1

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  Astar Island — Ensemble Prediction  (v5)               ║")
    print(f"║  Regression 72% + DilatedConvNet 28%  κ={BAYES_KAPPA:.0f}          ║")
    print(f"║  Expected score: ~83.5  (LOO benchmark ens_k10=83.35)  ║")
    print("╚══════════════════════════════════════════════════════════╝\n")

    # ── 1. Load history ────────────────────────────────────────────────────────
    records = load_history()
    if not records:
        print("ERROR: no history files found in ./history/  "
              "Run download_history.py first."); return
    rounds = sorted(set(r["round_number"] for r in records))
    print(f"History: {len(records)} files from {len(rounds)} rounds.  "
          f"Rounds: {rounds}\n")

    # ── 2. Build regression + train neural models ──────────────────────────────
    print("Building additive regression…")
    reg, reg_ms, reg_me, reg_ss, reg_se, reg_cal = build_regression(records)
    print(f"  Regression keys: {len(reg)}  "
          f"regime: surv={reg_ms:.3f}±{reg_ss:.3f}  exp={reg_me:.3f}±{reg_se:.3f}")

    rpr    = compute_regime_per_round(records)
    rstats = compute_regime_stats(rpr)
    print(f"  Regime stats for NN: surv={rstats['surv_mean']:.3f}±{rstats['surv_std']:.3f}  "
          f"exp={rstats['exp_mean']:.3f}±{rstats['exp_std']:.3f}")

    # Save regime stats for future use
    import json as json_mod
    with open(REG_STATS, 'w') as f:
        json_mod.dump(rstats, f)

    print(f"\nTraining neural model (device={DEVICE})…")
    model_dil = get_dilated_model(records, rstats)

    # ── 3. Active round ────────────────────────────────────────────────────────
    all_rounds = get_rounds()
    active = next((r for r in all_rounds if r["status"] == "active"), None)
    if not active:
        print("No active round found.")
        for r in all_rounds[-5:]:
            print(f"  Round {r['round_number']:3d}  {r['status']}  "
                  f"{r.get('closes_at', '?')}")
        return

    round_id       = active["id"]
    print(f"Round {active['round_number']}  |  closes: {active.get('closes_at', '?')}")
    detail         = get_round(round_id)
    N_seeds        = detail["seeds_count"]
    initial_states = detail["initial_states"]

    # Derive W and H from the actual grid (API map_width/height can differ from
    # the grid stored in initial_states, causing out-of-bounds at edge tiles)
    _g0 = initial_states[0]["grid"]
    H   = len(_g0)
    W   = len(_g0[0]) if H > 0 else detail["map_width"]
    print(f"Map: {W}×{H} (API reported {detail['map_width']}×{detail['map_height']})  "
          f"Seeds: {N_seeds}")

    budget_info       = get_budget()
    queries_used      = budget_info.get("queries_used", 0)
    queries_max       = budget_info.get("queries_max", 50)
    queries_remaining = queries_max - queries_used
    print(f"Budget: {queries_used}/{queries_max}\n")

    # ── 4. Full-map regime estimation (9 tiles × 5 seeds = 45 queries) ─────────
    tiles      = get_tiles(W, H, 15)       # tiles sized to ACTUAL grid dimensions
    n_tiles    = len(tiles)
    q_budget   = min(n_tiles * N_seeds, queries_remaining)
    q_per_seed = min(n_tiles, q_budget // N_seeds) if N_seeds > 0 else 0

    print(f"── Phase 1: Full-map regime estimation "
          f"({q_per_seed} tiles × {N_seeds} seeds = "
          f"{q_per_seed * N_seeds} queries) ──")

    combined      = {"s_surv": 0, "s_total": 0, "p_exp": 0, "p_total": 0}
    budget_spent  = 0
    # Per-seed per-cell observations for Bayesian update
    cell_obs = [defaultdict(list) for _ in range(N_seeds)]

    for seed_idx in range(N_seeds):
        print(f"  Seed {seed_idx}: ", end="", flush=True)
        init_grid = initial_states[seed_idx]["grid"]
        last_res  = None
        for i, (tx, ty) in enumerate(tiles[:q_per_seed]):
            if budget_spent >= queries_remaining:
                print(f"(budget exhausted at tile {i})", end=""); break
            try:
                res = simulate(round_id, seed_idx, tx, ty)
                grid = res["grid"]
                # Accumulate regime
                accumulate_regime(grid, tx, ty, init_grid, combined)
                # Accumulate per-cell observations
                for row_i, row in enumerate(grid):
                    for col_i, cell in enumerate(row):
                        cx, cy = tx + col_i, ty + row_i
                        if 0 <= cx < W and 0 <= cy < H:
                            cls = TERRAIN_TO_CLASS.get(cell, 0)
                            cell_obs[seed_idx][(cx, cy)].append(cls)
                budget_spent += 1; last_res = res
                print(".", end="", flush=True)
            except Exception as e:
                print(f"E({tx},{ty}:{e})", end="", flush=True)
            if budget_spent < queries_remaining:
                time.sleep(0.22)   # 5 req/sec limit
        used_str = last_res.get("queries_used", "?") if last_res else "?"
        print(f"  s={combined['s_surv']}/{combined['s_total']}  "
              f"p={combined['p_exp']}/{combined['p_total']}  "
              f"budget={used_str}/{queries_max}")
    n_obs_per_cell = q_per_seed  # each tile gives 1 obs/cell; tiles cover full map
    print(f"  Observations per cell: ~{n_obs_per_cell} (from {q_per_seed} tiles)")

    # Compute regime estimates
    obs_surv = combined["s_surv"] / combined["s_total"] if combined["s_total"] > 0 else reg_ms
    obs_exp  = combined["p_exp"]  / combined["p_total"] if combined["p_total"] > 0 else reg_me

    print(f"\nObserved regime (pooled across all {N_seeds} seeds × {q_per_seed} tiles):")
    print(f"  Settlement survival: {obs_surv:.3f}  (hist mean={reg_ms:.3f})")
    print(f"  Plains expansion:    {obs_exp:.3f}  (hist mean={reg_me:.3f})")

    # ── 5. Ensemble predict and submit ────────────────────────────────────────
    print(f"\n── Phase 2: Ensemble predict & submit "
          f"(reg_w={ENSEMBLE_W}, kappa={BAYES_KAPPA}) ──────────────────")
    for seed_idx in range(N_seeds):
        init_grid = initial_states[seed_idx]["grid"]

        # Regression prediction (already accounts for regime)
        p_reg = predict_regression(init_grid, W, H, reg, reg_ms, reg_me,
                                   reg_ss, reg_se, obs_surv, obs_exp,
                                   reg_cal, clip=CLIP_SIGMA)

        # DilatedConvNet (31×31 RF) — regime-standardised + output-smoothed
        p_dil = predict_map(model_dil, init_grid, W, H, obs_surv, obs_exp,
                            regime_stats=rstats, output_smooth=0.02)

        # Bayesian update: refine neural predictions using seed's observations
        # Only dynamic cells (high entropy) get meaningful updates;
        # static cells (ocean/mountain) have near-zero entropy and score nothing.
        p_dil_bay = bayesian_cell_update(p_dil, cell_obs[seed_idx],
                                         kappa=BAYES_KAPPA)

        # Final ensemble: stable regression + Bayesian-updated DilatedConvNet
        pred = blend_predictions(p_reg, p_dil_bay, ENSEMBLE_W)

        ent = -np.sum(np.maximum(pred, 1e-10) * np.log(np.maximum(pred, 1e-10)), axis=-1)
        print(f"  Seed {seed_idx}: avg_ent={ent.mean():.4f}  "
              f"min_p={pred.min():.4f}  submitting…", end=" ", flush=True)
        result = submit_seed(round_id, seed_idx, pred)
        print(result)
        time.sleep(0.6)   # 2 req/sec limit

    print(f"\n✓ Done!  Total queries used this run: {budget_spent}/{queries_max}")
    if os.path.exists(WEIGHTS_DL):
        print(f"  Model weights: {WEIGHTS_DL} ({os.path.getsize(WEIGHTS_DL)//1024} KB)")
    print()
    print("  Regime signal quality:")
    print(f"    Settlement cells observed: {combined['s_total']}  "
          f"(expected ~{40 * N_seeds})")
    print(f"    Plains cells observed:     {combined['p_total']}  "
          f"(expected ~{960 * N_seeds})")


if __name__ == "__main__":
    main()
