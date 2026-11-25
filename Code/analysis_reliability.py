import argparse
import csv
import itertools
import os

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score

import jconfig as C
from jmetrics import (best_f1_threshold, compute_metrics, cost_optimal_threshold, expected_cost,
                      precision_recall_at_budget, threshold_metrics)
from jutils import get_logger, read_json, run_dir_for

LOG = None
MAIN_METRICS = ["auc_prc", "auc_prc_lift", "auc_roc", "f1_fraud", "precision", "recall", "mcc", "balanced_accuracy",
                "f1_macro", "brier", "ece", "threshold"]


# ---------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------

_cache = {}


def load_run(exp, d, c, m, s):
    key = (exp, d, c, m, s)
    if key in _cache:
        return _cache[key]
    rd = run_dir_for(C.DIRS["runs"], exp, d, c, m, s)
    if not os.path.exists(os.path.join(rd, "DONE")):
        _cache[key] = None
        return None
    met = read_json(os.path.join(rd, "metrics.json"))
    z = np.load(os.path.join(rd, "predictions.npz"), allow_pickle=False)
    r = {k: z[k] for k in z.files}
    r["threshold"] = met["threshold"]
    r["metrics"] = met["metrics"]
    _cache[key] = r
    return r


def runs_of(exp, d, c, m):
    out = {}
    for s in C.EXPERIMENTS[exp]["seeds"]:
        r = load_run(exp, d, c, m, s)
        if r is not None:
            out[s] = r
    return out


def specs():
    """(exp, dataset, condition, model) for every defined experiment cell."""
    for exp, sp in C.EXPERIMENTS.items():
        for d in C.experiment_datasets(exp):
            for c in sp["conditions"]:
                for m in sp["models"]:
                    yield exp, d, c, m


def write_csv(name, rows):
    if not rows:
        LOG.info(f"{name}: no rows (runs missing?)")
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    p = os.path.join(C.DIRS["analysis"], f"{name}.csv")
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    LOG.info(f"Wrote {p} ({len(rows)} rows)")


def _agg(vals):
    v = np.asarray([x for x in vals if x is not None and not np.isnan(x)], dtype=float)
    if len(v) == 0:
        return {"mean": None, "sd": None, "cv": None, "min": None, "max": None, "median": None, "n": 0}
    sd = float(v.std(ddof=1)) if len(v) > 1 else 0.0
    return {"mean": float(v.mean()), "sd": sd, "cv": sd / abs(v.mean()) if v.mean() != 0 else None,
            "min": float(v.min()), "max": float(v.max()), "median": float(np.median(v)), "n": int(len(v))}


# ---------------------------------------------------------------------
# Analyses
# ---------------------------------------------------------------------

def seed_stability():
    rows = []
    for exp, d, c, m in specs():
        runs = runs_of(exp, d, c, m)
        if not runs:
            continue
        for metric in MAIN_METRICS:
            a = _agg([r["metrics"]["test"].get(metric) for r in runs.values()])
            gap = _agg([r["metrics"]["val"].get(metric, np.nan) - r["metrics"]["test"].get(metric, np.nan)
                        for r in runs.values()]) if metric != "threshold" else {"mean": None}
            rows.append({"experiment": exp, "dataset": d, "condition": c, "model": m, "metric": metric,
                         **{f"test_{k}": v for k, v in a.items()}, "val_minus_test_mean": gap["mean"]})
    write_csv("seed_stability", rows)


def ratio_stability():
    rows = []
    for d in C.DATASETS:
        for m in C.MAIN_MODELS:
            for metric in ("auc_prc", "f1_fraud", "mcc", "recall", "precision", "ece"):
                row = {"dataset": d, "model": m, "metric": metric}
                means = []
                for c in ("random60", "random70", "random80"):
                    exp = "E2" if c == "random70" else "E2b"
                    a = _agg([r["metrics"]["test"][metric] for r in runs_of(exp, d, c, m).values()])
                    row.update({f"{c}_mean": a["mean"], f"{c}_sd": a["sd"], f"{c}_n": a["n"]})
                    if a["mean"] is not None:
                        means.append(a["mean"])
                row["range_of_means"] = (max(means) - min(means)) if len(means) == 3 else None
                rows.append(row)
    write_csv("ratio_stability", rows)


def _common(ra, rb):
    ia = {n: i for i, n in enumerate(ra["test_node_id"])}
    idx_a, idx_b = [], []
    for j, n in enumerate(rb["test_node_id"]):
        if n in ia:
            idx_a.append(ia[n])
            idx_b.append(j)
    return np.asarray(idx_a, int), np.asarray(idx_b, int)


def churn():
    rows = []
    for exp, d, c, m in specs():
        if exp not in ("E2", "E2b"):
            continue
        runs = runs_of(exp, d, c, m)
        seeds = sorted(runs)
        if len(seeds) < 2:
            continue
        flips, rhos, budget_j = [], [], {b: [] for b in C.ALERT_BUDGETS}
        n_common = []
        for a, b in itertools.combinations(seeds, 2):
            ra, rb = runs[a], runs[b]
            ia, ib = _common(ra, rb)
            if len(ia) < 10:
                continue
            n_common.append(len(ia))
            pa, pb = ra["test_prob"][ia], rb["test_prob"][ib]
            flips.append(float(np.mean((pa >= ra["threshold"]) != (pb >= rb["threshold"]))))
            rhos.append(float(spearmanr(pa, pb).correlation))
            for bud in C.ALERT_BUDGETS:
                k = max(1, int(round(bud * len(ia))))
                ta = set(np.argsort(-pa, kind="stable")[:k].tolist())
                tb = set(np.argsort(-pb, kind="stable")[:k].tolist())
                budget_j[bud].append(len(ta & tb) / len(ta | tb))
        row = {"experiment": exp, "dataset": d, "condition": c, "model": m, "n_seeds": len(seeds),
               "n_pairs": len(flips), "mean_common_nodes": float(np.mean(n_common)) if n_common else None,
               "alert_flip_mean": float(np.mean(flips)) if flips else None,
               "alert_flip_sd": float(np.std(flips, ddof=1)) if len(flips) > 1 else None,
               "score_spearman_mean": float(np.mean(rhos)) if rhos else None}
        for bud in C.ALERT_BUDGETS:
            row[f"topk_jaccard_{bud}"] = float(np.mean(budget_j[bud])) if budget_j[bud] else None
        # consensus and union recall (same node set needed -> temporal only)
        if c == "temporal":
            ids0 = runs[seeds[0]]["test_node_id"]
            if all(np.array_equal(runs[s]["test_node_id"], ids0) for s in seeds):
                y = runs[seeds[0]]["test_y"] == 1
                A = np.stack([runs[s]["test_prob"] >= runs[s]["threshold"] for s in seeds])
                row["consensus_recall_all_seeds"] = float(A.all(0)[y].mean()) if y.any() else None
                row["union_recall_any_seed"] = float(A.any(0)[y].mean()) if y.any() else None
                row["mean_single_seed_recall"] = float(A[:, y].mean()) if y.any() else None
                row["nodes_flagged_by_some_not_all_share"] = float((A.any(0) & ~A.all(0)).mean())
        rows.append(row)
    write_csv("churn", rows)


def threshold_transfer():
    rows = []
    for exp, d, c, m in specs():
        for s, r in runs_of(exp, d, c, m).items():
            y, p = r["test_y"], r["test_prob"]
            t_val = r["threshold"]
            t_or = best_f1_threshold(y, p)
            a = threshold_metrics(y, p, t_val)
            b = threshold_metrics(y, p, t_or)
            rows.append({"experiment": exp, "dataset": d, "condition": c, "model": m, "seed": s,
                         "threshold_val": t_val, "threshold_test_oracle": t_or,
                         "f1_at_val_threshold": a["f1_fraud"], "f1_at_oracle_threshold": b["f1_fraud"],
                         "f1_gap": b["f1_fraud"] - a["f1_fraud"],
                         "alerts_at_val_threshold": a["n_alerts"], "n_fraud_test": int(y.sum()),
                         "alert_volume_ratio": a["n_alerts"] / max(int(y.sum()), 1),
                         "f1_at_0.5": threshold_metrics(y, p, 0.5)["f1_fraud"]})
    write_csv("threshold_transfer", rows)


def alert_budget():
    rows = []
    for exp, d, c, m in specs():
        for s, r in runs_of(exp, d, c, m).items():
            for b in C.ALERT_BUDGETS:
                pr = precision_recall_at_budget(r["test_y"], r["test_prob"], b)
                rows.append({"experiment": exp, "dataset": d, "condition": c, "model": m, "seed": s, "budget": b,
                             **pr, "prevalence": float(np.mean(r["test_y"]))})
    write_csv("alert_budget", rows)


def calibration():
    rows, bins_rows = [], []
    edges = np.linspace(0, 1, C.ECE_BINS + 1)
    for exp, d, c, m in specs():
        runs = runs_of(exp, d, c, m)
        if not runs:
            continue
        ys, ps = [], []
        for s, r in runs.items():
            t = r["metrics"]["test"]
            rows.append({"experiment": exp, "dataset": d, "condition": c, "model": m, "seed": s,
                         "ece": t["ece"], "brier": t["brier"], "mean_prob": float(r["test_prob"].mean()),
                         "prevalence": t["prevalence"]})
            ys.append(r["test_y"])
            ps.append(r["test_prob"])
        y, p = np.concatenate(ys).astype(float), np.concatenate(ps).astype(float)
        idx = np.clip(np.digitize(p, edges[1:-1]), 0, C.ECE_BINS - 1)
        for b in range(C.ECE_BINS):
            sel = idx == b
            bins_rows.append({"experiment": exp, "dataset": d, "condition": c, "model": m, "bin": b,
                              "bin_lo": edges[b], "bin_hi": edges[b + 1], "n": int(sel.sum()),
                              "mean_prob": float(p[sel].mean()) if sel.any() else None,
                              "fraud_rate": float(y[sel].mean()) if sel.any() else None})
    write_csv("calibration", rows)
    write_csv("calibration_bins", bins_rows)


def cost():
    rows = []
    for exp, d, c, m in specs():
        if exp not in ("E2", "E5"):
            continue
        for s, r in runs_of(exp, d, c, m).items():
            for ratio in C.COST_RATIOS:
                t_c = cost_optimal_threshold(r["val_y"], r["val_prob"], ratio)
                ec = expected_cost(r["test_y"], r["test_prob"], t_c, ratio)
                ef = expected_cost(r["test_y"], r["test_prob"], r["threshold"], ratio)
                t_or = cost_optimal_threshold(r["test_y"], r["test_prob"], ratio)
                eo = expected_cost(r["test_y"], r["test_prob"], t_or, ratio)
                rows.append({"experiment": exp, "dataset": d, "condition": c, "model": m, "seed": s,
                             "fn_fp_ratio": ratio, "threshold_cost_val": t_c,
                             "norm_cost_cost_threshold": ec["normalised_cost"],
                             "norm_cost_f1_threshold": ef["normalised_cost"],
                             "norm_cost_test_oracle": eo["normalised_cost"], "fn": ec["fn"], "fp": ec["fp"]})
    write_csv("cost", rows)


def complementarity():
    rows = []
    for d in C.DATASETS:
        for c in ("temporal", "random70"):
            models = C.MAIN_MODELS + ["XGBoost-raw", "KNN-raw"]
            for s in C.SEEDS_20:
                rs = {}
                for m in models:
                    exp = "E5" if m in C.RAW_MODELS else "E2"
                    r = load_run(exp, d, c, m, s)
                    if r is not None:
                        rs[m] = r
                for a, b in itertools.combinations(sorted(rs), 2):
                    ra, rb = rs[a], rs[b]
                    ia, ib = _common(ra, rb)
                    if len(ia) == 0:
                        continue
                    y = ra["test_y"][ia] == 1
                    pa = ra["test_prob"][ia] >= ra["threshold"]
                    pb = rb["test_prob"][ib] >= rb["threshold"]
                    ca, cb = pa == y, pb == y
                    n11, n00 = int((ca & cb).sum()), int((~ca & ~cb).sum())
                    n10, n01 = int((ca & ~cb).sum()), int((~ca & cb).sum())
                    den = n11 * n00 + n01 * n10
                    rows.append({"dataset": d, "condition": c, "seed": s, "model_a": a, "model_b": b,
                                 "n_nodes": int(len(ia)), "n_fraud": int(y.sum()),
                                 "disagreement": (n10 + n01) / len(ia),
                                 "q_statistic": (n11 * n00 - n01 * n10) / den if den else None,
                                 "fraud_caught_a_only": float((pa & ~pb)[y].mean()) if y.any() else None,
                                 "fraud_caught_b_only": float((~pa & pb)[y].mean()) if y.any() else None,
                                 "fraud_missed_both": float((~pa & ~pb)[y].mean()) if y.any() else None,
                                 "union_recall": float((pa | pb)[y].mean()) if y.any() else None,
                                 "false_alert_overlap_jaccard": float(((pa & pb) & ~y).sum() / max(((pa | pb) & ~y).sum(), 1)),
                                 "score_spearman": float(spearmanr(ra["test_prob"][ia], rb["test_prob"][ib]).correlation)})
    write_csv("complementarity", rows)


def _safe_ap(y, p):
    return float(average_precision_score(y, p)) if 0 < y.sum() < len(y) else None


def degree():
    rows = []
    for d in C.DATASETS:
        for c in ("temporal", "random70"):
            ref = load_run("E2", d, c, "GCN", C.SEEDS_20[0])
            if ref is None:
                continue
            q = np.unique(np.quantile(ref["test_degree"], [0, 0.25, 0.5, 0.75, 0.9, 1.0]))
            for m in C.MAIN_MODELS:
                for s, r in runs_of("E2", d, c, m).items():
                    b = np.clip(np.searchsorted(q, r["test_degree"], side="right") - 1, 0, len(q) - 2)
                    for i in range(len(q) - 1):
                        sel = b == i
                        y, p = r["test_y"][sel], r["test_prob"][sel]
                        if sel.sum() == 0:
                            continue
                        tm = threshold_metrics(y, p, r["threshold"])
                        rows.append({"dataset": d, "condition": c, "model": m, "seed": s, "degree_bin": i,
                                     "deg_lo": float(q[i]), "deg_hi": float(q[i + 1]), "n": int(sel.sum()),
                                     "n_fraud": int(y.sum()), "auc_prc": _safe_ap(y, p),
                                     "recall": tm["recall"], "precision": tm["precision"], "f1_fraud": tm["f1_fraud"]})
    write_csv("degree", rows)


def per_step():
    rows = []
    for d in C.DATASETS:
        for m in C.MAIN_MODELS:
            for s, r in runs_of("E2", d, "temporal", m).items():
                for t in np.unique(r["test_time"]):
                    sel = r["test_time"] == t
                    y, p = r["test_y"][sel], r["test_prob"][sel]
                    tm = threshold_metrics(y, p, r["threshold"])
                    rows.append({"dataset": d, "model": m, "seed": s, "time_step": int(t), "n": int(sel.sum()),
                                 "n_fraud": int(y.sum()), "auc_prc": _safe_ap(y, p), "f1_fraud": tm["f1_fraud"],
                                 "recall": tm["recall"], "precision": tm["precision"]})
    write_csv("per_step", rows)


def tuning_effect():
    rows = []
    for d in C.DATASETS:
        for c in ("temporal", "random70"):
            for m in C.MAIN_MODELS:
                row = {"dataset": d, "condition": c, "model": m}
                for exp in ("E2", "E3", "E4"):
                    if m not in C.EXPERIMENTS[exp]["models"]:
                        continue
                    for metric in ("auc_prc", "f1_fraud", "mcc"):
                        a = _agg([r["metrics"]["test"][metric] for r in runs_of(exp, d, c, m).values()])
                        row[f"{exp}_{metric}_mean"] = a["mean"]
                        row[f"{exp}_{metric}_sd"] = a["sd"]
                for metric in ("auc_prc", "f1_fraud", "mcc"):
                    for exp in ("E3", "E4"):
                        if row.get(f"E2_{metric}_mean") is not None and row.get(f"{exp}_{metric}_mean") is not None:
                            row[f"E2_minus_{exp}_{metric}"] = row[f"E2_{metric}_mean"] - row[f"{exp}_{metric}_mean"]
                rows.append(row)
    write_csv("tuning_effect", rows)


def tbns_k():

    rows = []
    sources = [(C.TBNS_K_MAIN, "E2", "ethereum")] + [(k, "E6K", d) for k, d in zip(C.TBNS_K_VARIANTS, C.K_DATASETS)]
    for c in ("temporal", "random70"):
        for m in C.MAIN_MODELS:
            for k, exp, d in sorted(sources):
                runs = [load_run(exp, d, c, m, s) for s in C.SEEDS_10]
                runs = [r for r in runs if r is not None]
                row = {"condition": c, "model": m, "tbns_k": k, "dataset": d, "experiment": exp, "n_seeds": len(runs)}
                for metric in ("auc_prc", "auc_prc_lift", "f1_fraud", "mcc", "recall", "precision", "ece"):
                    a = _agg([r["metrics"]["test"][metric] for r in runs])
                    row[f"{metric}_mean"], row[f"{metric}_sd"] = a["mean"], a["sd"]
                rows.append(row)
    write_csv("tbns_k_sensitivity", rows)


def recompute_check():

    worst, n = 0.0, 0
    for exp, d, c, m in specs():
        for _s, r in runs_of(exp, d, c, m).items():
            mm = compute_metrics(r["test_y"], r["test_prob"], r["threshold"], C.ECE_BINS)
            for k in ("auc_prc", "f1_fraud", "mcc"):
                a, b = mm[k], r["metrics"]["test"][k]
                if not (np.isnan(a) and np.isnan(b)):
                    worst = max(worst, abs(a - b))
            n += 1
    LOG.info(f"integrity: {n} runs recomputed, max |difference| vs metrics.json = {worst:.2e}")
    return worst


ANALYSES = {"seed_stability": seed_stability, "ratio_stability": ratio_stability, "churn": churn,
            "threshold_transfer": threshold_transfer, "alert_budget": alert_budget, "calibration": calibration,
            "cost": cost, "complementarity": complementarity, "degree": degree, "per_step": per_step,
            "tuning_effect": tuning_effect, "tbns_k": tbns_k, "recompute_check": recompute_check}


def main():
    global LOG
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", choices=list(ANALYSES))
    args, _ = ap.parse_known_args()
    C.ensure_dirs()
    LOG = get_logger("analysis_reliability", C.DIRS["logs"])
    for name, fn in ANALYSES.items():
        if args.only and name not in args.only:
            continue
        LOG.info(f"--- {name}")
        fn()


if __name__ == "__main__":
    main()
