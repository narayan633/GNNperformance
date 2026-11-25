"""
analysis_statistics.py  
"""

import argparse
import csv
import hashlib
import itertools
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.stats import friedmanchisquare, rankdata, wilcoxon

import jconfig as C
from jutils import get_logger, read_json, run_dir_for

LOG = None


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def holm(pvals):
    p = np.asarray([1.0 if v is None or np.isnan(v) else v for v in pvals], float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj.tolist()


def weighted_ap(y_sorted, w_sorted):

    tp = np.cumsum(w_sorted * y_sorted)
    fp = np.cumsum(w_sorted * (1 - y_sorted))
    prec = tp / np.maximum(tp + fp, 1e-12)
    pos = (w_sorted * y_sorted).sum()
    return float((w_sorted * y_sorted * prec).sum() / pos) if pos > 0 else np.nan


def load_test(exp, d, c, m, s):
    rd = run_dir_for(C.DIRS["runs"], exp, d, c, m, s)
    if not os.path.exists(os.path.join(rd, "DONE")):
        return None
    z = np.load(os.path.join(rd, "predictions.npz"), allow_pickle=False)
    met = read_json(os.path.join(rd, "metrics.json"))
    return {"id": z["test_node_id"], "y": z["test_y"].astype(np.int64), "p": z["test_prob"].astype(np.float64),
            "m": met["metrics"]["test"]}


def write_csv(name, rows):
    if not rows:
        LOG.info(f"{name}: no rows")
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


def cohens_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2:
        return None, None
    pooled = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    sd_diff = (a - b).std(ddof=1)
    d = float((a.mean() - b.mean()) / pooled) if pooled > 0 else None
    dz = float((a - b).mean() / sd_diff) if sd_diff > 0 else None
    return d, dz


def _effect_label(d):
    if d is None:
        return None
    x = abs(d)
    return "negligible" if x < 0.2 else "small" if x < 0.5 else "medium" if x < 0.8 else "large"


def _wilcoxon(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = ~(np.isnan(a) | np.isnan(b))
    a, b = a[ok], b[ok]
    d, dz = cohens_d(a, b)
    if len(a) < 5 or np.allclose(a, b):
        return {"n_pairs": int(len(a)), "mean_diff": float(np.mean(a - b)) if len(a) else None, "p": None,
                "cohens_d": d, "cohens_dz": dz, "cohens_d_label": _effect_label(d),
                "wins_a": int((a > b).sum()), "wins_b": int((b > a).sum())}
    st = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
    diff = a - b
    # rank-biserial effect size
    r = rankdata(np.abs(diff[diff != 0]))
    rb = (r[diff[diff != 0] > 0].sum() - r[diff[diff != 0] < 0].sum()) / r.sum() if r.sum() else 0.0
    return {"n_pairs": int(len(a)), "mean_diff": float(diff.mean()), "median_diff": float(np.median(diff)),
            "W": float(st.statistic), "p": float(st.pvalue), "rank_biserial": float(rb),
            "cohens_d": d, "cohens_dz": dz, "cohens_d_label": _effect_label(d),
            "wins_a": int((a > b).sum()), "wins_b": int((b > a).sum())}


# ---------------------------------------------------------------------
# Paired bootstrap
# ---------------------------------------------------------------------

def _bootstrap_pair(args):
    d, c, a, b, seeds, n_boot = args
    per_seed = []
    for s in seeds:
        ra, rb = load_test("E2", d, c, a, s), load_test("E2", d, c, b, s)
        if ra is None or rb is None:
            continue
        ib_map = {n: i for i, n in enumerate(rb["id"])}
        ia = np.array([i for i, n in enumerate(ra["id"]) if n in ib_map], int)
        ib = np.array([ib_map[ra["id"][i]] for i in ia], int)
        if len(ia) == 0:
            continue
        y = ra["y"][ia]
        oa = np.argsort(-ra["p"][ia], kind="stable")
        ob = np.argsort(-rb["p"][ib], kind="stable")
        per_seed.append((y, oa, ob, len(ia), s))
    if not per_seed:
        return None
    rng = np.random.default_rng(int(hashlib.md5(f"{d}|{c}|{a}|{b}".encode()).hexdigest(), 16) % (2 ** 32))
    obs = np.mean([weighted_ap(y[oa], np.ones(n)[oa]) - weighted_ap(y[ob], np.ones(n)[ob])
                   for y, oa, ob, n, _ in per_seed])
    diffs = np.empty(n_boot)
    for bi in range(n_boot):
        acc = 0.0
        for y, oa, ob, n, _ in per_seed:
            w = np.bincount(rng.integers(0, n, n), minlength=n).astype(np.float64)
            acc += weighted_ap(y[oa], w[oa]) - weighted_ap(y[ob], w[ob])
        diffs[bi] = acc / len(per_seed)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return {"dataset": d, "condition": c, "model_a": a, "model_b": b, "n_seeds": len(per_seed),
            "boot_ap_diff": float(obs), "boot_ci_lo": float(lo), "boot_ci_hi": float(hi),
            "boot_p": float(min(1.0, max(p, 1.0 / n_boot)))}


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------

def pairwise_models(n_boot, workers, boot_seeds):
    rows = []
    jobs = []
    for d in C.DATASETS:
        for c in ("temporal", "random70"):
            runs = {m: {s: load_test("E2", d, c, m, s) for s in C.SEEDS_20} for m in C.MAIN_MODELS}
            for a, b in itertools.combinations(C.MAIN_MODELS, 2):
                for metric in ("auc_prc", "f1_fraud", "mcc", "f1_macro"):
                    sa = [runs[a][s]["m"][metric] if runs[a][s] else np.nan for s in C.SEEDS_20]
                    sb = [runs[b][s]["m"][metric] if runs[b][s] else np.nan for s in C.SEEDS_20]
                    rows.append({"dataset": d, "condition": c, "metric": metric, "model_a": a, "model_b": b,
                                 "mean_a": float(np.nanmean(sa)) if not np.all(np.isnan(sa)) else None,
                                 "mean_b": float(np.nanmean(sb)) if not np.all(np.isnan(sb)) else None,
                                 **_wilcoxon(sa, sb)})
                jobs.append((d, c, a, b, boot_seeds, n_boot))
    # Holm within dataset x condition x metric
    for key in {(r["dataset"], r["condition"], r["metric"]) for r in rows}:
        grp = [r for r in rows if (r["dataset"], r["condition"], r["metric"]) == key]
        for r, adj in zip(grp, holm([r["p"] for r in grp])):
            r["p_holm"] = adj
    if n_boot > 0:
        LOG.info(f"bootstrap: {len(jobs)} model pairs, {n_boot} resamples, {len(boot_seeds)} seeds, {workers} workers")
        with ProcessPoolExecutor(max_workers=workers) as ex:
            boots = [b for b in ex.map(_bootstrap_pair, jobs) if b is not None]
        for key in {(b["dataset"], b["condition"]) for b in boots}:
            grp = [b for b in boots if (b["dataset"], b["condition"]) == key]
            for b, adj in zip(grp, holm([b["boot_p"] for b in grp])):
                b["boot_p_holm"] = adj
        bmap = {(b["dataset"], b["condition"], b["model_a"], b["model_b"]): b for b in boots}
        for r in rows:
            if r["metric"] == "auc_prc":
                b = bmap.get((r["dataset"], r["condition"], r["model_a"], r["model_b"]))
                if b:
                    r.update({k: v for k, v in b.items() if k.startswith("boot")})
    write_csv("stats_pairwise_models", rows)


def _paired_exp(exp_a, exp_b, d, c, ma, mb, seeds, metric):
    sa, sb = [], []
    for s in seeds:
        ra, rb = load_test(exp_a, d, c, ma, s), load_test(exp_b, d, c, mb, s)
        sa.append(ra["m"][metric] if ra else np.nan)
        sb.append(rb["m"][metric] if rb else np.nan)
    return {"mean_a": float(np.nanmean(sa)) if not np.all(np.isnan(sa)) else None,
            "mean_b": float(np.nanmean(sb)) if not np.all(np.isnan(sb)) else None, **_wilcoxon(sa, sb)}


def tuning():
    rows = []
    for d in C.DATASETS:
        for c in ("temporal", "random70"):
            for metric in ("auc_prc", "f1_fraud", "mcc", "f1_macro"):
                fam = []
                for m in C.ICBC_MODELS:
                    fam.append({"family": "tuned_vs_fixed", "dataset": d, "condition": c, "metric": metric,
                                "model": m, "a": "E2 tuned", "b": "E3 ICBC fixed",
                                **_paired_exp("E2", "E3", d, c, m, m, C.SEEDS_20, metric)})
                for r, adj in zip(fam, holm([r["p"] for r in fam])):
                    r["p_holm"] = adj
                rows += fam
                fam = []
                for m in C.MAIN_MODELS:
                    fam.append({"family": "tuned_vs_transferred", "dataset": d, "condition": c, "metric": metric,
                                "model": m, "a": "E2 own-condition settings",
                                "b": f"E4 settings tuned on {C.transfer_condition_for(c)}",
                                **_paired_exp("E2", "E4", d, c, m, m, C.SEEDS_10, metric)})
                for r, adj in zip(fam, holm([r["p"] for r in fam])):
                    r["p_holm"] = adj
                rows += fam
    write_csv("stats_tuning", rows)


ABLATION_FAMILIES = {
    "component": [("GCN+XGBoost", "E2", "XGBoost-raw", "E5"), ("GCN+KNN", "E2", "KNN-raw", "E5"),
                  ("GNN+Autoencoder", "E2", "GNN+Autoencoder-noRecon", "E5")],
    "encoder_control": [("GCN+XGBoost", "E2", "GCN+XGBoost-noEdge", "E5"),
                        ("GCN+XGBoost-noEdge", "E5", "XGBoost-raw", "E5")],
}


def ablation():
    rows = []
    for d in C.DATASETS:
        for c in ("temporal", "random70"):
            for metric in ("auc_prc", "f1_fraud", "mcc", "f1_macro"):
                for family, pairs in ABLATION_FAMILIES.items():
                    fam = []
                    for ma, ea, mb, eb in pairs:
                        fam.append({"family": family, "dataset": d, "condition": c, "metric": metric,
                                    "model_a": ma, "model_b": mb,
                                    **_paired_exp(ea, eb, d, c, ma, mb, C.SEEDS_20, metric)})
                    for r, adj in zip(fam, holm([r["p"] for r in fam])):
                        r["p_holm"] = adj
                    rows += fam
    write_csv("stats_ablation", rows)


ENCODER_CONTROL_MODELS = ["XGBoost-raw", "GCN+XGBoost-noEdge", "GCN+XGBoost"]


def encoder_control():

    rob = _read_csv(os.path.join(C.DIRS["robustness"], "summary.csv"))
    if not rob:
        LOG.warning("encoder_control: robustness/summary.csv missing")
        return
    per = {}
    edge_check = {}
    for r in rob:
        m = r["model"]
        if m not in ENCODER_CONTROL_MODELS:
            continue
        key = (r["dataset"], r["condition"], m, r["kind"], float(r["level"]))
        per.setdefault(key, {}).setdefault(int(r["seed"]), []).append(float(r["delta_auc_prc"]))
        if m in C.NOEDGE_MODELS and r["kind"] == "edge_drop":
            k2 = (r["dataset"], r["condition"])
            edge_check[k2] = max(edge_check.get(k2, 0.0), abs(float(r["delta_auc_prc"])))
    rows = []
    for d in C.DATASETS:
        for c in ("temporal", "random70"):
            for lv in C.FEATURE_NOISE_LEVELS:
                vec = {}
                for m in ENCODER_CONTROL_MODELS:
                    seeds = per.get((d, c, m, "feature_noise", float(lv)), {})
                    vec[m] = [float(np.mean(seeds[s])) if s in seeds else np.nan for s in C.SEEDS_10]
                base = {"dataset": d, "condition": c, "noise_level": lv,
                        "noedge_edge_drop_max_abs_delta": edge_check.get((d, c))}
                for m in ENCODER_CONTROL_MODELS:
                    v = np.asarray(vec[m], float)
                    rows.append({**base, "row": "model", "model": m,
                                 "mean_delta_auc_prc": float(np.nanmean(v)) if np.isfinite(v).any() else None,
                                 "sd_delta_auc_prc": float(np.nanstd(v, ddof=1)) if np.isfinite(v).sum() > 1 else None,
                                 "n_seeds": int(np.isfinite(v).sum())})
                fam = []
                for a, b in (("GCN+XGBoost-noEdge", "XGBoost-raw"), ("GCN+XGBoost", "GCN+XGBoost-noEdge"),
                             ("GCN+XGBoost", "XGBoost-raw")):
                    fam.append({**base, "row": "pair", "model": f"{a} vs {b}", **_wilcoxon(vec[a], vec[b])})
                for r, adj in zip(fam, holm([r["p"] for r in fam])):
                    r["p_holm"] = adj
                rows += fam
    write_csv("stats_encoder_control", rows)


def tbns_k():
    """K = 1 and K = 5 against K = 3 (E2 ethereum), paired by seed over SEEDS_10, Holm within family."""
    rows = []
    for c in ("temporal", "random70"):
        for metric in ("auc_prc", "f1_fraud", "mcc", "f1_macro"):
            for k, d in zip(C.TBNS_K_VARIANTS, C.K_DATASETS):
                fam = []
                for m in C.MAIN_MODELS:
                    sa, sb = [], []
                    for s in C.SEEDS_10:
                        ra, rb = load_test("E6K", d, c, m, s), load_test("E2", "ethereum", c, m, s)
                        sa.append(ra["m"][metric] if ra else np.nan)
                        sb.append(rb["m"][metric] if rb else np.nan)
                    fam.append({"condition": c, "metric": metric, "model": m, "a": f"K={k}",
                                "b": f"K={C.TBNS_K_MAIN}",
                                "mean_a": float(np.nanmean(sa)) if not np.all(np.isnan(sa)) else None,
                                "mean_b": float(np.nanmean(sb)) if not np.all(np.isnan(sb)) else None,
                                **_wilcoxon(sa, sb)})
                for r, adj in zip(fam, holm([r["p"] for r in fam])):
                    r["p_holm"] = adj
                rows += fam
    write_csv("stats_tbns_k", rows)


def ranks():
    rows = []
    for d in C.DATASETS:
        for c in ("temporal", "random70", "random60", "random80"):
            exp = "E2b" if c in ("random60", "random80") else "E2"
            for metric in ("auc_prc", "f1_fraud", "mcc", "f1_macro"):
                M = np.full((len(C.SEEDS_20), len(C.MAIN_MODELS)), np.nan)
                for i, s in enumerate(C.SEEDS_20):
                    for j, m in enumerate(C.MAIN_MODELS):
                        r = load_test(exp, d, c, m, s)
                        if r:
                            M[i, j] = r["m"][metric]
                M = M[~np.isnan(M).any(1)]
                if len(M) < 3:
                    continue
                R = np.vstack([rankdata(-row) for row in M])
                fr = friedmanchisquare(*M.T)
                for j, m in enumerate(C.MAIN_MODELS):
                    rows.append({"dataset": d, "condition": c, "metric": metric, "model": m,
                                 "mean_rank": float(R[:, j].mean()), "n_seeds": int(len(M)),
                                 "friedman_chi2": float(fr.statistic), "friedman_df": int(M.shape[1] - 1),
                                 "friedman_p": float(fr.pvalue),
                                 "friedman_scope": "omnibus test within one dataset x condition, seeds as blocks"})
    write_csv("stats_ranks", rows)




def main():
    global LOG
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--n-boot", type=int, default=C.BOOTSTRAP_N)
    ap.add_argument("--boot-seeds", type=int, default=20, help="number of seeds used in the node bootstrap")
    ap.add_argument("--only", nargs="*", choices=["pairwise", "tuning", "ablation", "encoder_control", "tbns_k", "ranks", "synthesis"])
    args, _ = ap.parse_known_args()
    C.ensure_dirs()
    LOG = get_logger("analysis_statistics", C.DIRS["logs"])
    todo = args.only or ["pairwise", "tuning", "ablation", "encoder_control", "tbns_k", "ranks", "synthesis"]
    if "pairwise" in todo:
        pairwise_models(args.n_boot, args.workers, C.SEEDS_20[:args.boot_seeds])
    if "tuning" in todo:
        tuning()
    if "ablation" in todo:
        ablation()
    if "encoder_control" in todo:
        encoder_control()
    if "tbns_k" in todo:
        tbns_k()
    if "ranks" in todo:
        ranks()
    if "synthesis" in todo:
        synthesis()


if __name__ == "__main__":
    main()
