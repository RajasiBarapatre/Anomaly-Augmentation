"""
Calibrated evaluator

Per Fold:
  - Training anomalies are clustered (K-means, 8 pseudo-types, one-hot + standardised)
  - Each fold hides 2 whole clusters from the generator (simulated unseen types) and holds out 30% of normals and 30% of the remaining (seen) anomalies
  - 'seen'   = held-out normals + held-out seen anomalies at prevalence 0.3
  - 'unseen' = held-out normals + hidden-cluster anomalies at prevalence 0.3 -> guardrail for the private test

Commands to run:
  python calibrated_proxy.py --dataset unsw            # IForest 3 seeds x 200 trees
  python calibrated_proxy.py --dataset unsw --full     # 10 seeds x 500 trees
"""
import argparse
import contextlib
import io
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score as ap

import final_augmentation_pipeline as pipe
from revised_experiment_config import DATASETS

warnings.filterwarnings("ignore")
LABEL = "is_anomaly"


def encode(train, test, feats, cats):
    xs, xt = [], []
    for c in feats:
        if c in cats:
            lv = np.asarray(sorted(train[c].astype(str).unique()), dtype=object)
            xs.append((train[c].astype(str).to_numpy(dtype=object)[:, None] == lv[None]).astype(float))
            xt.append((test[c].astype(str).to_numpy(dtype=object)[:, None] == lv[None]).astype(float))
        else:
            xs.append(train[[c]].to_numpy(float))
            xt.append(test[[c]].to_numpy(float))
    Xs, Xt = np.hstack(xs), np.hstack(xt)
    mu, sd = Xs.mean(0), Xs.std(0)
    sd[sd == 0] = 1.0
    return (Xs - mu) / sd, (Xt - mu) / sd


def make_folds(d, feats, cats, n_folds=4, k=8, seed=0):
    rng = np.random.default_rng(seed)
    A = d[d[LABEL] == 1]
    Xa, _ = encode(A, A, feats, cats)
    types = KMeans(k, n_init=5, random_state=seed).fit_predict(Xa)
    order = rng.permutation(k)
    out = []
    for f in range(n_folds):
        hidden = order[2 * f: 2 * f + 2]
        idx_n = d.index[d[LABEL] == 0].to_numpy().copy()
        rng.shuffle(idx_n)
        ho_n = idx_n[: int(0.3 * len(idx_n))]
        unseen = A.index[np.isin(types, hidden)].to_numpy()
        seen = A.index[~np.isin(types, hidden)].to_numpy().copy()
        rng.shuffle(seen)
        ho_seen = seen[: int(0.3 * len(seen))]
        train = d.drop(np.concatenate([ho_n, unseen, ho_seen]))
        test = d.loc[np.concatenate([ho_n, ho_seen, unseen])]
        grp = np.r_[np.zeros(len(ho_n)), np.ones(len(ho_seen)), np.full(len(unseen), 2)]
        out.append((train, test, grp))
    return out


def detector_scores(syn, test, feats, cats, seeds, trees):
    from pyod.models.ecod import ECOD
    Xs, Xt = encode(syn, test, feats, cats)
    e = ECOD().fit(Xs).decision_function(Xt)
    i = np.mean([-IsolationForest(n_estimators=trees, random_state=s, n_jobs=-1).fit(Xs).score_samples(Xt)
                 for s in range(seeds)], 0)
    return e, i


def subset_score(e, i, grp, which, prev=0.3, reps=10):
    rng = np.random.default_rng(0)
    n0, pool = np.flatnonzero(grp == 0), np.flatnonzero(grp == which)
    na = min(int(prev / (1 - prev) * len(n0)), len(pool))
    vals = []
    for _ in range(reps):
        m = np.r_[n0, rng.choice(pool, na, replace=False)]
        y = (grp[m] > 0).astype(int)
        vals.append(((ap(y, e[m]) + ap(y, i[m])) / 2, ap(y, e[m]), ap(y, i[m])))
    return np.mean(vals, 0)


def generate(train, feats, key, seed):
    normal = train.loc[train[LABEL] == 0, feats].reset_index(drop=True)
    attack = train.loc[train[LABEL] == 1, feats].reset_index(drop=True)
    with contextlib.redirect_stdout(io.StringIO()):
        syn, _ = pipe.generate_synthetic_normals(
            normal, attack, target_n=DATASETS[key]["target_n"], random_state=seed,
            candidate_over_sample_factor=pipe.CANDIDATE_OVER_SAMPLE_FACTORS[key],
            realism_weight=pipe.REALISM_WEIGHTS[key],
            mixture_adapt_strength=pipe.MIXTURE_ADAPT_STRENGTHS[key],
            attack_overlap_percentile=pipe.ATTACK_OVERLAP_PERCENTILES[key],
            use_tvae=pipe.USE_TVAE[key], coverage_filter=pipe.COVERAGE_FILTER[key],
            empirical_share=pipe.EMPIRICAL_SHARES[key])
    return syn[feats]


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--dataset", choices=["unsw", "nslkdd"], default="unsw")
    ap_.add_argument("--full", action="store_true", help="IForest 10 seeds x 500 trees")
    ap_.add_argument("--seed", type=int, default=42)
    args = ap_.parse_args()
    cfg = DATASETS[args.dataset]
    d = pd.read_csv(cfg["path"], keep_default_na=False, na_values=[""])
    feats = [c for c in d.columns if c not in (LABEL, "id")]
    cats = cfg["categorical_cols"]
    seeds, trees = (10, 500) if args.full else (3, 200)
    rows = []
    for f, (tr, te, grp) in enumerate(make_folds(d, feats, cats)):
        t = time.time()
        syn = generate(tr, feats, args.dataset, args.seed)
        e, i = detector_scores(syn, te[feats], feats, cats, seeds, trees)
        s, se, si = subset_score(e, i, grp, 1)
        u = subset_score(e, i, grp, 2)[0]
        rows.append(dict(seen=s, ecod=se, iforest=si, unseen=u))
        print(f"  fold {f + 1}: seen S={s:.4f} (ECOD {se:.4f}, IForest {si:.4f})  unseen S={u:.4f}  "
              f"[{time.time() - t:.0f}s]", flush=True)
    r = pd.DataFrame(rows)
    print(f"\n{args.dataset}: seen S = {r.seen.mean():.4f} +/- {r.seen.std():.4f}   "
          f"unseen S = {r.unseen.mean():.4f} +/- {r.unseen.std():.4f}")


if __name__ == "__main__":
    main()
