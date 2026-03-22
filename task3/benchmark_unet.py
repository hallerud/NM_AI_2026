#!/usr/bin/env python3
"""
benchmark_unet.py  —  Leave-one-round-out benchmark (v3)
=========================================================

Models tested:
  1. Additive regression (v4 baseline)
  2. ConvNet  (7×7 RF,  ch=64, 3 layers, ~81K params)
  3. DilatedConvNet (31×31 RF, ch=64, dilations=[1,2,4,8], ~105K params)

Each neural model is also tested with Bayesian cell update:
  - 5 observations per cell (across 5 seeds × full-map tiles)
  - Dirichlet-Multinomial posterior: q_k = (κ·p_k + count_k) / (κ + 5)
  - Tested at κ ∈ {5, 10, 20}

Final ensemble: best neural (dilated) + regression blend.

Key insight from scoring: ONLY DYNAMIC CELLS (high GT entropy) are scored.
Bayesian update is most valuable there — the posterior from 5 samples
outperforms the model wherever the model is miscalibrated.

GPU:  Automatically uses CUDA (set in unet.py via DEVICE).
      Expected runtime on RTX 5080: ~40–70s per fold → ~14–21 min total.
"""

import json, os, sys, time
import numpy as np
from glob import glob
from collections import defaultdict, deque

import torch
print(f"CUDA available: {torch.cuda.is_available()}")

sys.path.insert(0, os.path.dirname(__file__))
from unet import (UNet, ConvNet, DilatedConvNet,
                  train_model, predict_map, build_dataset,
                  compute_regime_per_round, compute_regime_stats,
                  bayesian_cell_update, get_dynamic_tiles,
                  blend_predictions, score_from_pred_gt,
                  N_CLASSES, EPS, _settle_distances, DEVICE)

HISTORY_DIR = "history"
N_TRIALS    = 20     # stochastic regime trials per fold
EPOCHS      = 200    # training epochs per fold
BATCH       = 64

# Bayesian prior strengths to test
KAPPAS = [5.0, 10.0, 20.0]

# ── Shared helpers ────────────────────────────────────────────────────────────

def apply_floor(pred):
    pred = np.maximum(pred, EPS)
    return pred / pred.sum(axis=-1, keepdims=True)

def score(pred, gt):
    return score_from_pred_gt(pred, gt)

def _is_coastal_arr(ig, W, H):
    coast = np.zeros((H, W), dtype=bool)
    for y in range(H):
        for x in range(W):
            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                nx, ny = x+dx, y+dy
                if nx<0 or ny<0 or nx>=W or ny>=H or ig[ny][nx]==10:
                    coast[y,x] = True; break
    return coast

def dist_bucket(d):
    if d==0: return 0
    if d<=2: return 1
    if d<=4: return 2
    if d<=6: return 3
    if d<=9: return 4
    return 5

# ── Additive regression (v4, LOO) ────────────────────────────────────────────

def build_additive_reg_loo(records, test_rn):
    train = [r for r in records if r["round_number"] != test_rn]
    rr_d = defaultdict(list)
    for rec in train:
        rn,gt,ig,H,W = rec["round_number"],rec["ground_truth"],rec["initial_grid"],rec["H"],rec["W"]
        ss=st=pe=pt=0
        for y in range(H):
            for x in range(W):
                c=ig[y][x]
                if c==1:   ss+=float(gt[y,x,1]+gt[y,x,2]); st+=1
                elif c==11: pe+=float(gt[y,x,1]+gt[y,x,2]+gt[y,x,3]); pt+=1
        rr_d[rn].append((ss/st if st else 0, pe/pt if pt else 0))
    rr={rn:np.mean(v,axis=0) for rn,v in rr_d.items()}
    ms=float(np.mean([v[0] for v in rr.values()]))
    me=float(np.mean([v[1] for v in rr.values()]))
    ss2=float(np.std([v[0] for v in rr.values()])+1e-6)
    se2=float(np.std([v[1] for v in rr.values()])+1e-6)
    tr=defaultdict(lambda:defaultdict(list))
    for rec in train:
        rn,gt,ig,H,W=rec["round_number"],rec["ground_truth"],rec["initial_grid"],rec["H"],rec["W"]
        dist=_settle_distances(ig,W,H)
        acc=defaultdict(lambda:np.zeros(N_CLASSES,dtype=np.float64)); cnt=defaultdict(int)
        for y in range(H):
            for x in range(W):
                key=(ig[y][x],dist_bucket(int(dist[y,x])),bool(_is_coastal_arr(ig,W,H)[y,x]))
                acc[key]+=gt[y,x]; cnt[key]+=1
        for k in acc: tr[k][rn].append(acc[k]/cnt[k])
    reg={}
    for key,rd in tr.items():
        rounds=sorted(rd.keys()); probs=np.array([np.mean(rd[rn],axis=0) for rn in rounds])
        base=probs.mean(axis=0)
        if len(rounds)<6: reg[key]=(base.astype(np.float32),None,None); continue
        sv=np.array([rr[rn][0]-ms for rn in rounds]); ev=np.array([rr[rn][1]-me for rn in rounds])
        X=np.column_stack([sv,ev])
        try:
            beta=np.linalg.lstsq(X,probs-base,rcond=None)[0]
            reg[key]=(base.astype(np.float32),beta[0].astype(np.float32),beta[1].astype(np.float32))
        except: reg[key]=(base.astype(np.float32),None,None)
    acc2=defaultdict(lambda:np.zeros(N_CLASSES)); cnt2=defaultdict(int)
    for rec in train:
        gt,ig,H,W=rec["ground_truth"],rec["initial_grid"],rec["H"],rec["W"]
        for y in range(H):
            for x in range(W): acc2[ig[y][x]]+=gt[y,x]; cnt2[ig[y][x]]+=1
    cal={code:(lambda p:p/p.sum())((lambda p:np.maximum(p,EPS))(acc2[code]/cnt2[code])).astype(np.float32) for code in acc2}
    return reg,ms,me,ss2,se2,cal

def predict_additive(ig,W,H,reg,ms,me,ss,se,obs_s,obs_e,cal,clip=3.5):
    sd=float(np.clip(obs_s-ms,-clip*ss,clip*ss)); ed=float(np.clip(obs_e-me,-clip*se,clip*se))
    dist=_settle_distances(ig,W,H); pred=np.zeros((H,W,N_CLASSES),dtype=np.float32)
    coast=_is_coastal_arr(ig,W,H)
    for y in range(H):
        for x in range(W):
            key=(ig[y][x],dist_bucket(int(dist[y,x])),bool(coast[y,x]))
            if key in reg:
                base,bs,be=reg[key]; p=base.copy()
                if bs is not None: p=p+bs*sd+be*ed
            elif ig[y][x] in cal: p=cal[ig[y][x]].copy()
            else: p=np.full(N_CLASSES,1/N_CLASSES,dtype=np.float32)
            p=np.maximum(p,0); s=p.sum()
            pred[y,x]=p/s if s>1e-10 else np.full(N_CLASSES,1/N_CLASSES)
    return apply_floor(pred)

# ── Regime + observation accumulation ────────────────────────────────────────

def sample_obs_full(gt, vx, vy, vw, vh, rng):
    """Sample one observation per cell; return dict (x,y)→class."""
    obs = {}
    for y in range(vy, min(vy+vh, gt.shape[0])):
        for x in range(vx, min(vx+vw, gt.shape[1])):
            p = gt[y,x].astype(np.float64)
            p = np.maximum(p, 0); p /= p.sum()
            cls = int(rng.choice(N_CLASSES, p=p))
            obs[(x, y)] = cls
    return obs

def accum_regime(obs_dict, ig):
    ss=st=pe=pt=0
    for (x,y), cls in obs_dict.items():
        c = ig[y][x]
        if c==1:
            if cls in (1,2): ss+=1
            st+=1
        elif c==11:
            if cls in (1,2,3): pe+=1
            pt+=1
    return ss, st, pe, pt

def get_tiles(W, H, sz=15):
    tiles, x = [], 0
    while x < W:
        y = 0
        while y < H: tiles.append((x,y)); y+=sz
        x += sz
    return tiles

# ── Main benchmark ────────────────────────────────────────────────────────────

def run():
    print("=" * 110)
    print("  Astar Island — Multi-Model Benchmark  (LOO, v3 with Bayesian update)")
    print("=" * 110)

    files   = sorted(glob(os.path.join(HISTORY_DIR, "round_*_seed_*.json")))
    records = []
    for path in files:
        with open(path) as f: rec = json.load(f)
        records.append({"round_number":rec["round_number"],"seed_index":rec["seed_index"],
                        "W":rec["map_width"],"H":rec["map_height"],
                        "initial_grid":rec["initial_grid"],
                        "ground_truth":np.array(rec["ground_truth"],dtype=np.float32)})
    rounds = sorted(set(r["round_number"] for r in records))
    print(f"\nHistory: {len(records)} files, {len(rounds)} rounds.  "
          f"Device: {DEVICE}  Epochs={EPOCHS}  Trials={N_TRIALS}  "
          f"Kappas={KAPPAS}\n")

    rng = np.random.default_rng(42)

    # Column layout: model × (oracle, 9q-plain, 9q-bay_k5, 9q-bay_k10, 9q-bay_k20)
    # Plus a final ensemble column
    cols = [
        "reg_orc", "reg_9q",
        "dil_orc", "dil_9q", "dil_k5", "dil_k10", "dil_k20",
        "ens_orc", "ens_k10",   # 50/50 reg+dilated, oracle and Bayesian k=10
    ]
    print(f"{'Rnd':>4}  " + "  ".join(f"{c:>9}" for c in cols))
    print("-" * (6 + 11 * len(cols)))

    all_scores = {c: [] for c in cols}

    for test_rn in rounds:
        t0 = time.time()
        test_recs  = [r for r in records if r["round_number"] == test_rn]
        train_recs = [r for r in records if r["round_number"] != test_rn]
        H, W = test_recs[0]["H"], test_recs[0]["W"]
        tiles = get_tiles(W, H, 15)

        # ── Additive regression (LOO) ────────────────────────────────────
        add_reg, add_ms, add_me, add_ss, add_se, add_cal = \
            build_additive_reg_loo(records, test_rn)

        # Oracle regime from GT (pooled across test seeds)
        oss=ost=ope=opt=0
        for rec in test_recs:
            gt,ig=rec["ground_truth"],rec["initial_grid"]
            for y in range(H):
                for x in range(W):
                    c=ig[y][x]
                    if c==1:   oss+=float(gt[y,x,1]+gt[y,x,2]); ost+=1
                    elif c==11: ope+=float(gt[y,x,1]+gt[y,x,2]+gt[y,x,3]); opt+=1
        or_s=oss/ost if ost else add_ms; or_e=ope/opt if opt else add_me

        # Regression oracle
        reg_oracle = []
        for rec in test_recs:
            p = predict_additive(rec["initial_grid"],W,H,add_reg,add_ms,add_me,
                                 add_ss,add_se,or_s,or_e,add_cal,99.0)
            reg_oracle.append(score(p, rec["ground_truth"]))

        # ── Train neural models ──────────────────────────────────────────
        rpr   = compute_regime_per_round(train_recs)
        stats = compute_regime_stats(rpr)
        dataset = build_dataset(train_recs, rpr, use_augmentation=True,
                               regime_stats=stats, regime_noise_std=0.3)

        # DilatedConvNet (primary neural model — 31×31 RF)
        model_dil = train_model(dataset, epochs=EPOCHS, lr=3e-3, batch=BATCH,
                                dropout=0.1, label_smooth=0.05,
                                model_type="dilated", convnet_ch=64,
                                verbose=False)

        # ── Oracle predictions (perfect regime, no observation noise) ───
        dil_orc  = []; ens_orc = []
        for rec in test_recs:
            ig = rec["initial_grid"]; gt = rec["ground_truth"]
            p_reg = predict_additive(ig,W,H,add_reg,add_ms,add_me,
                                     add_ss,add_se,or_s,or_e,add_cal,99.0)
            p_dil = predict_map(model_dil, ig, W, H, or_s, or_e,
                               regime_stats=stats, output_smooth=0.02)
            dil_orc.append(score(p_dil, gt))
            ens_orc.append(score(blend_predictions(p_reg, p_dil, 0.5), gt))

        # ── Stochastic 9q×5s trials ──────────────────────────────────────
        # For each trial: sample full map across all seeds,
        # accumulate (a) regime counters and (b) per-cell observation lists

        per_seed_reg  = [[] for _ in test_recs]
        per_seed_dil  = [[] for _ in test_recs]
        per_seed_dil_k = {k: [[] for _ in test_recs] for k in KAPPAS}
        per_seed_ens_k10 = [[] for _ in test_recs]

        for _ in range(N_TRIALS):
            # Accumulate regime + per-cell observations across all seeds
            reg_ctrs = [0, 0, 0, 0]
            # cell_obs[i] = dict (x,y)→list[class] for seed i
            cell_obs = [defaultdict(list) for _ in test_recs]

            for i, rec in enumerate(test_recs):
                for tx, ty in tiles:
                    obs_dict = sample_obs_full(rec["ground_truth"], tx, ty, 15, 15, rng)
                    # Update regime
                    a, b, c, d = accum_regime(obs_dict, rec["initial_grid"])
                    reg_ctrs[0]+=a; reg_ctrs[1]+=b; reg_ctrs[2]+=c; reg_ctrs[3]+=d
                    # Store per-cell observations
                    for (x,y), cls in obs_dict.items():
                        cell_obs[i][(x,y)].append(cls)

            obs_s = reg_ctrs[0]/reg_ctrs[1] if reg_ctrs[1]>0 else add_ms
            obs_e = reg_ctrs[2]/reg_ctrs[3] if reg_ctrs[3]>0 else add_me

            for i, rec in enumerate(test_recs):
                ig = rec["initial_grid"]; gt = rec["ground_truth"]

                # Base predictions
                p_reg = predict_additive(ig,W,H,add_reg,add_ms,add_me,
                                        add_ss,add_se,obs_s,obs_e,add_cal,3.5)
                p_dil = predict_map(model_dil, ig, W, H, obs_s, obs_e,
                                   regime_stats=stats, output_smooth=0.02)

                per_seed_reg[i].append(score(p_reg, gt))
                per_seed_dil[i].append(score(p_dil, gt))

                # Bayesian-updated neural predictions at each kappa
                for kap in KAPPAS:
                    p_bay = bayesian_cell_update(p_dil, cell_obs[i], kappa=kap)
                    per_seed_dil_k[kap][i].append(score(p_bay, gt))

                # Ensemble: reg + Bayesian-dilated (k=10)
                p_bay10 = bayesian_cell_update(p_dil, cell_obs[i], kappa=10.0)
                p_ens   = blend_predictions(p_reg, p_bay10, 0.5)
                per_seed_ens_k10[i].append(score(p_ens, gt))

        # ── Aggregate ────────────────────────────────────────────────────
        def avg(lst_of_lists):
            return float(np.mean([np.mean(s) for s in lst_of_lists]))

        row = {
            "reg_orc": np.mean(reg_oracle),
            "reg_9q":  avg(per_seed_reg),
            "dil_orc": np.mean(dil_orc),
            "dil_9q":  avg(per_seed_dil),
            "dil_k5":  avg(per_seed_dil_k[5.0]),
            "dil_k10": avg(per_seed_dil_k[10.0]),
            "dil_k20": avg(per_seed_dil_k[20.0]),
            "ens_orc": np.mean(ens_orc),
            "ens_k10": avg(per_seed_ens_k10),
        }
        for c in cols:
            all_scores[c].append(row[c])

        elapsed = time.time() - t0
        print(f"R{test_rn:2d}   " +
              "  ".join(f"{row[c]:9.1f}" for c in cols) +
              f"  [{elapsed:.0f}s]")

    # ── Summary ──────────────────────────────────────────────────────────
    print("-" * (6 + 11 * len(cols)))
    print("AVG  " + "  ".join(f"{np.mean(all_scores[c]):9.1f}" for c in cols))
    print()

    # Best kappa analysis
    print("Bayesian kappa analysis (DilatedConvNet 9q×5s):")
    for rn_i, rn in enumerate(rounds):
        base  = all_scores["dil_9q"][rn_i]
        k5    = all_scores["dil_k5"][rn_i]
        k10   = all_scores["dil_k10"][rn_i]
        k20   = all_scores["dil_k20"][rn_i]
        ens   = all_scores["ens_k10"][rn_i]
        print(f"  R{rn:2d}  no-bay={base:5.1f}  k5={k5:5.1f}  "
              f"k10={k10:5.1f}  k20={k20:5.1f}  ens={ens:5.1f}")
    print()
    print("Averages:")
    for c in cols:
        print(f"  {c:>10}: {np.mean(all_scores[c]):.2f}")


if __name__ == "__main__":
    run()
