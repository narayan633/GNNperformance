"""
perturbation_robustness.py

"""

import argparse
import copy
import csv
import os

import numpy as np
from scipy.stats import spearmanr

import jconfig as C
from jmetrics import compute_metrics
from jutils import release_gpu, RunLock, get_logger, now, read_json, run_dir_for, set_seed, write_json

LOG = None
ROBUST_SEEDS = C.SEEDS_10
ROBUST_CONDITIONS = ["temporal", "random70"]
ROBUST_MODELS = C.MAIN_MODELS + ["XGBoost-raw"] + C.NOEDGE_MODELS


def _perturb_data(d, kind, level, rng, train_sd):
    import torch
    d = copy.copy(d)
    if kind == "edge_drop":
        e = d.edge_index.size(1)
        keep = torch.as_tensor(rng.random(e) >= level, device=d.edge_index.device)
        d.edge_index = d.edge_index[:, keep]
    elif kind == "feature_noise":
        noise = torch.as_tensor(rng.standard_normal(tuple(d.x.shape)).astype(np.float32), device=d.x.device)
        noise = noise * torch.as_tensor(train_sd * level, dtype=torch.float32, device=d.x.device)
        noise[~d.has_features] = 0.0
        d.x = d.x + noise
    return d


def _perturb_inp(inp, kind, level, rng, train_sd, condition, is_seq):
    out = dict(inp)
    out["data"] = {k: (list(v) if isinstance(v, list) else v) for k, v in inp["data"].items()}
    if is_seq:
        key = "test" if condition == "temporal" else "full"
        out["data"][key] = [_perturb_data(d, kind, level, rng, train_sd) for d in inp["data"][key]]
    else:
        if condition == "temporal":
            out["data"]["test"] = _perturb_data(inp["data"]["test"], kind, level, rng, train_sd)
        else:
            d = _perturb_data(inp["data"]["test"], kind, level, rng, train_sd)
            out["data"] = {"train": d, "val": d, "test": d}
    return out


def _predict(model_name, kind, obj, inp, is_seq):
    from jtrain import _predict_flat_torch, _predict_hybrid, _predict_sequence
    if is_seq:
        return _predict_sequence(obj, inp, "test")
    if kind == "raw":
        d, m = inp["data"]["test"], inp["masks"]["test"]
        mm = m.cpu().numpy()
        p = obj.predict_proba(d.x.cpu().numpy()[mm])[:, 1]
        return p, d.y.cpu().numpy()[mm]
    if kind == "hybrid":
        p, mm = _predict_hybrid(obj, inp, "test")
    else:
        p, mm = _predict_flat_torch(obj, inp, "test")
    return p, inp["data"]["test"].y.cpu().numpy()[mm]


def _train_sd(inp, is_seq):
    import torch
    if is_seq:
        xs = [d.x[m] for d, m in zip(inp["data"]["train"] if "train" in inp["data"] else inp["data"]["full"],
                                     inp["masks"]["train"])]
        x = torch.cat(xs)
    else:
        x = inp["data"]["train"].x[inp["masks"]["train"]]
    sd = x.std(0).cpu().numpy()
    sd[sd < 1e-8] = 1.0
    return sd


def run_one_model(dataset, condition, model, seed, bundle, device):
    from jtrain import load_model, prepare_flat, prepare_sequence
    exp = C.source_experiment(model)
    rd = run_dir_for(C.DIRS["runs"], exp, dataset, condition, model, seed)
    met = read_json(os.path.join(rd, "metrics.json"))
    if met is None or not os.path.exists(os.path.join(rd, "DONE")):
        LOG.warning(f"missing run {rd}")
        return None
    set_seed(seed)
    is_seq = model in C.SEQUENCE_MODELS
    inp = prepare_sequence(bundle, condition, seed, device) if is_seq else prepare_flat(bundle, condition, seed, device)
    kind, obj = load_model(rd, device)
    thr = met["threshold"]
    p0, y0 = _predict(model, kind, obj, inp, is_seq)
    saved = np.load(os.path.join(rd, "predictions.npz"))
    consistency = float(np.max(np.abs(np.sort(p0) - np.sort(saved["test_prob"])))) if len(p0) == len(saved["test_prob"]) else None
    clean = compute_metrics(y0, p0, thr, C.ECE_BINS)
    a0 = p0 >= thr
    sd = _train_sd(inp, is_seq)
    out = {"dataset": dataset, "condition": condition, "model": model, "seed": seed, "source_run": rd,
           "threshold": thr, "clean": clean, "reload_max_abs_diff_vs_saved": consistency, "time": now(),
           "perturbations": []}
    tests = [("feature_noise", lv) for lv in C.FEATURE_NOISE_LEVELS]
    if kind != "raw":
        tests = [("edge_drop", r) for r in C.EDGE_DROP_RATES] + tests
    for kind_p, level in tests:
        for rep in range(C.ROBUSTNESS_REPEATS):
            rng = np.random.default_rng([seed, rep, int(level * 1e4), 0 if kind_p == "edge_drop" else 1])
            pin = _perturb_inp(inp, kind_p, level, rng, sd, condition, is_seq)
            p, y = _predict(model, kind, obj, pin, is_seq)
            m = compute_metrics(y, p, thr, C.ECE_BINS)
            out["perturbations"].append({
                "kind": kind_p, "level": level, "repeat": rep,
                "auc_prc": m["auc_prc"], "f1_fraud": m["f1_fraud"], "recall": m["recall"], "precision": m["precision"],
                "delta_auc_prc": m["auc_prc"] - clean["auc_prc"], "delta_f1": m["f1_fraud"] - clean["f1_fraud"],
                "alert_flip_share": float(np.mean((p >= thr) != a0)),
                "fraud_recall_lost_share": float(np.mean(a0[y0 == 1] & ~(p >= thr)[y0 == 1])) if (y0 == 1).any() else None,
                "spearman_vs_clean": float(spearmanr(p0, p).correlation)})
    return out


def write_summary():
    rows = []
    for root, _, files in os.walk(C.DIRS["robustness"]):
        for fn in files:
            if not fn.startswith("seed") or not fn.endswith(".json"):
                continue
            r = read_json(os.path.join(root, fn))
            for p in r["perturbations"]:
                rows.append({"dataset": r["dataset"], "condition": r["condition"], "model": r["model"],
                             "seed": r["seed"], "clean_auc_prc": r["clean"]["auc_prc"],
                             "clean_f1": r["clean"]["f1_fraud"], **p})
    if rows:
        path = os.path.join(C.DIRS["robustness"], "summary.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        LOG.info(f"Wrote {path} ({len(rows)} rows)")


def main():
    global LOG
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", nargs="*", choices=C.DATASETS, default=C.DATASETS)
    ap.add_argument("--model", nargs="*", choices=ROBUST_MODELS, default=ROBUST_MODELS)
    ap.add_argument("--condition", nargs="*", choices=ROBUST_CONDITIONS, default=ROBUST_CONDITIONS)
    ap.add_argument("--device", default="cuda")
    args, _ = ap.parse_known_args()
    C.ensure_dirs()
    LOG = get_logger("robustness", C.DIRS["logs"])
    import torch
    from jdata import DatasetBundle
    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    for d in args.dataset:
        bundle = DatasetBundle(d)
        for c in args.condition:
            for m in args.model:
                for s in ROBUST_SEEDS:
                    od = os.path.join(C.DIRS["robustness"], d, c, m.replace("+", "_plus_"))
                    out = os.path.join(od, f"seed{s}.json")
                    if os.path.exists(out):
                        continue
                    with RunLock(os.path.join(od, f"lock_seed{s}")) as lk:
                        if not lk.acquired or os.path.exists(out):
                            continue
                        try:
                            r = run_one_model(d, c, m, s, bundle, device)
                        except Exception as e:
                            LOG.exception(f"{d} {c} {m} seed{s} failed: {e}")
                            continue
                        if r is not None:
                            write_json(out, r)
                            LOG.info(f"{d} {c} {m} seed{s} done")
                            release_gpu(device)
    write_summary()


if __name__ == "__main__":
    main()
