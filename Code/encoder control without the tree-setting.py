import argparse
import csv
import json
import os
import time

import numpy as np

import jconfig as C
from jutils import (RunLock, get_logger, is_done, mark_done, now, read_json, release_gpu, run_dir_for,
                    stable_hash, write_json)

LOG = None
ROOT_OUT = os.path.join(C.OUT, "c1_matched_control")
CONDITIONS = ["temporal", "random70"]
VARIANTS = ["matched", "noCW"]


# ---------------------------------------------------------------------
# Settings of each variant
# ---------------------------------------------------------------------

def variant_params(variant, dataset, condition):
    """Returns the RawBaseline params of a variant, or None if the variant does not apply."""
    from params import params_for
    if variant == "matched":
        p, _ = params_for("E5", "GCN+XGBoost-noEdge", dataset, condition)
        src = "GCN+XGBoost tuned settings (as used by the no-edge pipeline)"
    elif variant == "noCW":
        p, _ = params_for("E5", "XGBoost-raw", dataset, condition)
        if not p.get("scale_pos_weight_on"):
            return None, None          # identical to tuned raw XGBoost
        src = "tuned raw XGBoost settings with class weighting switched off"
    else:
        raise ValueError(variant)
    out = {"xgb_n_estimators": int(p["xgb_n_estimators"]), "xgb_max_depth": int(p["xgb_max_depth"]),
           "xgb_learning_rate": float(p["xgb_learning_rate"]), "scale_pos_weight_on": False}
    return out, src


# ---------------------------------------------------------------------
# Clean runs (same code path as run_experiments.py)
# ---------------------------------------------------------------------

def clean_dir(variant, dataset, condition, seed):
    return os.path.join(ROOT_OUT, "runs", variant, dataset, condition, f"seed{seed}")


def run_clean(variant, dataset, condition, seed, bundle, device):
    from jtrain import run_one
    rd = clean_dir(variant, dataset, condition, seed)
    if is_done(rd):
        return "done"
    params, src = variant_params(variant, dataset, condition)
    if params is None:
        return "n/a"
    with RunLock(rd) as lock:
        if not lock.acquired or is_done(rd):
            return "locked"
        t0 = time.perf_counter()
        res = run_one("XGBoost-raw", bundle, condition, seed, params, device, run_dir=rd)
        res.update({"experiment": "C1", "variant": variant, "param_source": src, "params_hash": stable_hash(params),
                    "wall_s": time.perf_counter() - t0, "finished": now()})
        write_json(os.path.join(rd, "metrics.json"), res)
        mark_done(rd, {"wall_s": res["wall_s"]})
        t = res["metrics"]["test"]
        LOG.info(f"clean {variant} {dataset} {condition} seed{seed}  AUC-PRC {t['auc_prc']:.4f}")
        return "ok"


# ---------------------------------------------------------------------
# Feature noise
# ---------------------------------------------------------------------

def run_noise(variant, dataset, condition, seed, bundle, device):
    import perturbation_robustness as PR
    from jmetrics import compute_metrics
    from jtrain import load_model, prepare_flat
    from jutils import set_seed
    from scipy.stats import spearmanr
    out = os.path.join(ROOT_OUT, "noise", variant, dataset, condition, f"seed{seed}.json")
    if os.path.exists(out):
        return "done"
    rd = clean_dir(variant, dataset, condition, seed)
    if not is_done(rd):
        return "no clean run"
    met = read_json(os.path.join(rd, "metrics.json"))
    set_seed(seed)
    inp = prepare_flat(bundle, condition, seed, device)
    kind, obj = load_model(rd, device)
    assert kind == "raw"
    thr = met["threshold"]
    p0, y0 = PR._predict("XGBoost-raw", kind, obj, inp, False)
    clean = compute_metrics(y0, p0, thr, C.ECE_BINS)
    a0 = p0 >= thr
    sd = PR._train_sd(inp, False)
    rows = []
    for level in C.FEATURE_NOISE_LEVELS:
        for rep in range(C.ROBUSTNESS_REPEATS):
            rng = np.random.default_rng([seed, rep, int(level * 1e4), 1])    # same draw as the main study
            pin = PR._perturb_inp(inp, "feature_noise", level, rng, sd, condition, False)
            p, y = PR._predict("XGBoost-raw", kind, obj, pin, False)
            m = compute_metrics(y, p, thr, C.ECE_BINS)
            rows.append({"kind": "feature_noise", "level": level, "repeat": rep, "auc_prc": m["auc_prc"],
                         "f1_fraud": m["f1_fraud"], "delta_auc_prc": m["auc_prc"] - clean["auc_prc"],
                         "delta_f1": m["f1_fraud"] - clean["f1_fraud"],
                         "alert_flip_share": float(np.mean((p >= thr) != a0)),
                         "spearman_vs_clean": float(spearmanr(p0, p).correlation)})
    write_json(out, {"variant": variant, "dataset": dataset, "condition": condition, "seed": seed,
                     "threshold": thr, "clean": clean, "perturbations": rows, "time": now()})
    LOG.info(f"noise {variant} {dataset} {condition} seed{seed} done")
    return "ok"


# ---------------------------------------------------------------------
# Tables and tests
# ---------------------------------------------------------------------

def _holm(p):
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    adj = np.empty(len(p))
    run = 0.0
    for i, j in enumerate(order):
        run = max(run, (len(p) - i) * p[j])
        adj[j] = min(1.0, run)
    return adj


def _wilcoxon(a, b):
    from scipy.stats import wilcoxon
    d = np.asarray(a) - np.asarray(b)
    if np.allclose(d, 0):
        return {"W": None, "p": 1.0}
    st = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
    return {"W": float(st.statistic), "p": float(st.pvalue)}


def _rank_biserial(a, b):
    from scipy.stats import rankdata
    d = np.asarray(a) - np.asarray(b)
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    r = rankdata(np.abs(d))
    return float((r[d > 0].sum() - r[d < 0].sum()) / r.sum())


def _write(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    LOG.info(f"Wrote {path} ({len(rows)} rows)")


def summarise():
    import pandas as pd
    # clean metrics: C1 variants plus the existing tuned raw XGBoost and no-edge runs
    clean = []
    refs = {"XGBoost-raw (tuned)": ("E5", "XGBoost-raw"), "GCN+XGBoost-noEdge": ("E5", "GCN+XGBoost-noEdge"),
            "GCN+XGBoost": ("E2", "GCN+XGBoost")}
    for d in C.DATASETS:
        for c in CONDITIONS:
            for s in C.SEEDS_20:
                for v in VARIANTS:
                    r = read_json(os.path.join(clean_dir(v, d, c, s), "metrics.json"))
                    if r:
                        clean.append({"dataset": d, "condition": c, "model": f"XGBoost-raw ({v})", "seed": s,
                                      **{k: r["metrics"]["test"][k] for k in ("auc_prc", "f1_fraud", "mcc", "ece", "brier")}})
                for name, (exp, m) in refs.items():
                    r = read_json(os.path.join(run_dir_for(C.DIRS["runs"], exp, d, c, m, s), "metrics.json"))
                    if r:
                        clean.append({"dataset": d, "condition": c, "model": name, "seed": s,
                                      **{k: r["metrics"]["test"][k] for k in ("auc_prc", "f1_fraud", "mcc", "ece", "brier")}})
    _write(os.path.join(ROOT_OUT, "c1_clean.csv"), clean)


    noise = []
    for v in VARIANTS:
        for d in C.DATASETS:
            for c in CONDITIONS:
                for s in C.SEEDS_10:
                    r = read_json(os.path.join(ROOT_OUT, "noise", v, d, c, f"seed{s}.json"))
                    if r:
                        for p in r["perturbations"]:
                            noise.append({"dataset": d, "condition": c, "model": f"XGBoost-raw ({v})", "seed": s,
                                          "clean_auc_prc": r["clean"]["auc_prc"], "level": p["level"],
                                          "repeat": p["repeat"], "auc_prc": p["auc_prc"],
                                          "delta_auc_prc": p["delta_auc_prc"]})
    main = os.path.join(C.DIRS["robustness"], "summary.csv")
    if os.path.exists(main):
        m = pd.read_csv(main)
        m = m[(m.kind == "feature_noise") & m.model.isin(["XGBoost-raw", "GCN+XGBoost-noEdge", "GCN+XGBoost"])]
        m = m.assign(model=m.model.replace({"XGBoost-raw": "XGBoost-raw (tuned)"}))
        for r in m.itertuples(index=False):
            noise.append({"dataset": r.dataset, "condition": r.condition, "model": r.model, "seed": int(r.seed),
                          "clean_auc_prc": r.clean_auc_prc, "level": r.level, "repeat": int(r.repeat),
                          "auc_prc": r.auc_prc, "delta_auc_prc": r.delta_auc_prc})
    else:
        LOG.warning(f"{main} not found, comparisons with the main models are skipped")
    _write(os.path.join(ROOT_OUT, "c1_noise.csv"), noise)
    if not noise:
        return

    cl = pd.DataFrame(clean)
    nz = pd.DataFrame(noise)
    summ = []
    for (d, c, mo), g in cl.groupby(["dataset", "condition", "model"]):
        row = {"dataset": d, "condition": c, "model": mo, "clean_n_seeds": len(g),
               "clean_auc_prc_mean": g.auc_prc.mean(), "clean_auc_prc_sd": g.auc_prc.std(ddof=1)}
        for lv in C.FEATURE_NOISE_LEVELS:
            h = nz[(nz.dataset == d) & (nz.condition == c) & (nz.model == mo) & (nz.level == lv)]
            if len(h):
                sm = h.groupby("seed").delta_auc_prc.mean()
                row[f"loss_{lv}_mean"] = -sm.mean()
                row[f"loss_{lv}_sd"] = sm.std(ddof=1)
                row[f"auc_prc_under_{lv}"] = h.groupby("seed").auc_prc.mean().mean()
        summ.append(row)
    _write(os.path.join(ROOT_OUT, "c1_summary.csv"), summ)

    # paired tests on the loss (seed means of the 5 draws), Holm within dataset, split and level
    pairs = [("XGBoost-raw (matched)", "GCN+XGBoost-noEdge"),
             ("XGBoost-raw (matched)", "XGBoost-raw (tuned)"),
             ("XGBoost-raw (noCW)", "XGBoost-raw (tuned)")]
    tests = []
    for d in C.DATASETS:
        for c in CONDITIONS:
            for lv in C.FEATURE_NOISE_LEVELS:
                fam = []
                for a, b in pairs:
                    h = nz[(nz.dataset == d) & (nz.condition == c) & (nz.level == lv)]
                    sa = h[h.model == a].groupby("seed").delta_auc_prc.mean()
                    sb = h[h.model == b].groupby("seed").delta_auc_prc.mean()
                    common = sorted(set(sa.index) & set(sb.index))
                    if len(common) < 5:
                        continue
                    la, lb = -sa.loc[common].values, -sb.loc[common].values     # losses, larger = worse
                    fam.append({"dataset": d, "condition": c, "level": lv, "a": a, "b": b, "n_pairs": len(common),
                                "loss_a": la.mean(), "loss_b": lb.mean(), "mean_diff": (la - lb).mean(),
                                "median_diff": float(np.median(la - lb)), **_wilcoxon(la, lb),
                                "rank_biserial": _rank_biserial(la, lb),
                                "wins_a_lower_loss": int((la < lb).sum())})
                if fam:
                    for row, ph in zip(fam, _holm([r["p"] for r in fam])):
                        row["p_holm"] = float(ph)
                    tests += fam
    _write(os.path.join(ROOT_OUT, "c1_tests.csv"), tests)


# ---------------------------------------------------------------------

def main():
    global LOG
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", nargs="*", choices=C.DATASETS, default=C.DATASETS)
    ap.add_argument("--condition", nargs="*", choices=CONDITIONS, default=CONDITIONS)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--summarise", action="store_true")
    args, _ = ap.parse_known_args()
    C.ensure_dirs()
    os.makedirs(ROOT_OUT, exist_ok=True)
    LOG = get_logger("c1_matched_control", C.DIRS["logs"])
    if args.summarise:
        summarise()
        return
    import torch
    from jdata import DatasetBundle
    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    t_start = time.perf_counter()
    for d in args.dataset:
        bundle = DatasetBundle(d)
        for c in args.condition:
            for v in VARIANTS:
                params, src = variant_params(v, d, c)
                if params is None:
                    LOG.info(f"{v} {d} {c}: not applicable (tuned raw XGBoost has no class weighting)")
                    continue
                LOG.info(f"{v} {d} {c}: {json.dumps(params)}  ({src})")
                for s in C.SEEDS_20:
                    try:
                        run_clean(v, d, c, s, bundle, device)
                    except Exception as e:
                        LOG.exception(f"clean {v} {d} {c} seed{s} failed: {e}")
                for s in C.SEEDS_10:
                    try:
                        run_noise(v, d, c, s, bundle, device)
                    except Exception as e:
                        LOG.exception(f"noise {v} {d} {c} seed{s} failed: {e}")
                release_gpu(device)
        del bundle
    summarise()


if __name__ == "__main__":
    main()
