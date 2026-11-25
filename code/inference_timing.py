"""
inference_timing.py 
==============================
"""

import argparse
import csv
import os
import time

import numpy as np

import jconfig as C
from jutils import environment_info, get_logger, now, read_json, run_dir_for, write_json

LOG = None
TIMING_SEEDS = C.SEEDS_20[:3]
KNN_TRAIN_FRACTIONS = [0.25, 0.50, 0.75, 1.00]
TIMED_MODELS = C.MAIN_MODELS + ["XGBoost-raw", "KNN-raw"]


def _source_run(dataset, model, seed):
    exp = "E5" if model in C.RAW_MODELS else "E2"
    return run_dir_for(C.DIRS["runs"], exp, dataset, "random70", model, seed)


def _subset(n, frac, rng):
    if frac >= 1.0:
        return np.arange(n)
    return np.sort(rng.choice(n, size=max(1, int(round(frac * n))), replace=False))


def _induce(g, keep):
    mask = np.zeros(g["x"].shape[0], bool)
    mask[keep] = True
    remap = -np.ones(len(mask), np.int64)
    remap[keep] = np.arange(len(keep))
    ei = g["edge_index"]
    sel = mask[ei[0]] & mask[ei[1]]
    out = {k: (v[keep] if isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == len(mask) else v)
           for k, v in g.items() if k not in ("edge_index", "edge_count", "edge_last_ts")}
    out["edge_index"] = remap[ei[:, sel]]
    return out


def _apply_norm(g, run_dir):
    p = os.path.join(run_dir, "normalisation.npz")
    if not os.path.exists(p):
        return g
    z = np.load(p)
    g = dict(g)
    xn = ((g["x"] - z["mu"]) / z["sd"]).astype(np.float32)
    xn[~g["has_features"]] = 0.0
    g["x"] = xn
    return g


class _RssSampler:


    def __init__(self, interval=0.005):
        import threading
        try:
            import psutil
            self.proc = psutil.Process()
        except ImportError:
            self.proc = None
        self.interval = interval
        self.peak = 0
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, self.proc.memory_info().rss)
            self._stop.wait(self.interval)

    def __enter__(self):
        if self.proc is not None:
            self.base = self.proc.memory_info().rss
            self.peak = self.base
            self._t.start()
        return self

    def __exit__(self, *exc):
        if self.proc is not None:
            self._stop.set()
            self._t.join()
            self.peak = max(self.peak, self.proc.memory_info().rss)
        return False


def _timeit(fn, device, warmup, runs):

    import torch
    cuda = str(device).startswith("cuda")
    for _ in range(warmup):
        fn()
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    ts = []
    with _RssSampler() as mem:
        for _ in range(runs):
            if cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            fn()
            if cuda:
                torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
    ts = np.array(ts)
    mean, sd = float(ts.mean()), float(ts.std(ddof=1)) if len(ts) > 1 else 0.0
    return {"mean_s": mean, "sd_s": sd, "cv_pct": 100 * sd / mean if mean > 0 else None,
            "median_s": float(np.median(ts)), "q1_s": float(np.percentile(ts, 25)), "q3_s": float(np.percentile(ts, 75)),
            "min_s": float(ts.min()), "max_s": float(ts.max()), "all_s": ts.tolist(),
            "peak_gpu_mb": float(torch.cuda.max_memory_allocated() / 2 ** 20) if cuda else None,
            "peak_process_rss_gb": mem.peak / 2 ** 30 if mem.proc is not None else None,
            "process_rss_before_gb": mem.base / 2 ** 30 if mem.proc is not None else None}


def _n_params(kind, obj):
    if kind == "torch":
        return int(sum(p.numel() for p in obj.parameters()))
    if kind == "hybrid":
        return int(sum(p.numel() for p in obj.gcn.parameters()))
    return None


def time_model(dataset, model, seed, bundle, devices, warmup, runs):
    import torch
    from jdata import to_data
    from jtrain import load_model
    rd = _source_run(dataset, model, seed)
    if not os.path.exists(os.path.join(rd, "DONE")):
        LOG.warning(f"no trained run at {rd}, skipping")
        return None
    rng = np.random.default_rng(seed)
    full = _apply_norm(bundle.full, rd)
    n = full["x"].shape[0]
    subsets = {f: _subset(n, f, rng) for f in C.TIMING_FRACTIONS}
    res = {"dataset": dataset, "model": model, "seed": seed, "source_run": rd, "time": now(),
           "n_nodes_full": int(n), "n_edges_full": int(full["edge_index"].shape[1]),
           "model_files_bytes": {f: os.path.getsize(os.path.join(rd, f)) for f in ("model.pt", "classifier.pkl")
                                 if os.path.exists(os.path.join(rd, f))},
           "results": []}
    for dev in devices:
        kind, obj = load_model(rd, dev)
        res["n_params"] = _n_params(kind, obj)
        if model in C.SEQUENCE_MODELS:
            snaps_raw = bundle.snapshots("full")
        for f, keep in subsets.items():
            g = _induce(full, keep)
            if model in C.SEQUENCE_MODELS:
                keep_ids = set(np.asarray(full["node_id"]).astype(str)[keep].tolist())
                seq = []
                for sg in snaps_raw:
                    sg = _apply_norm(sg, rd)
                    k = np.nonzero(np.fromiter((i in keep_ids for i in np.asarray(sg["node_id"]).astype(str)),
                                               bool, count=len(sg["node_id"])))[0]
                    if len(k):
                        seq.append(to_data(_induce(sg, k), dev))
                n_scored = int(sum(d.num_nodes for d in seq))

                def fn(seq=seq):
                    with torch.no_grad():
                        return [torch.exp(o)[:, 1].cpu() for o in obj.run_sequence(seq)]
            elif kind == "raw":
                X = g["x"]
                n_scored = X.shape[0]

                def fn(X=X):
                    return obj.predict_proba(X)
            else:
                d = to_data(g, dev)
                n_scored = d.num_nodes
                if kind == "hybrid":
                    def fn(d=d):
                        return obj.predict_proba_emb(obj.embeddings(d))
                else:
                    def fn(d=d):
                        with torch.no_grad():
                            return torch.exp(obj(d.x, d.edge_index))[:, 1].cpu()
            t = _timeit(fn, dev, warmup, runs)
            t.update({"device": dev, "fraction": f, "n_nodes": int(len(keep)), "n_scored": n_scored,
                      "n_edges": int(g["edge_index"].shape[1]),
                      "nodes_per_s": n_scored / t["mean_s"] if t["mean_s"] > 0 else None,
                      "nodes_per_s_median": n_scored / t["median_s"] if t["median_s"] > 0 else None})
            res["results"].append(t)
            LOG.info(f"{dataset} {model} seed{seed} {dev} f={f:.2f} median {t['median_s'] * 1e3:.2f} ms")
            del fn
            if str(dev).startswith("cuda"):
                torch.cuda.empty_cache()
        if model == "GCN+KNN":
            res["knn_train_size"] = _knn_train_size(obj, full, rd, dev, warmup, runs, rng)
    return res


def _knn_train_size(hyb, full, rd, dev, warmup, runs, rng):

    from jdata import to_data
    pred = np.load(os.path.join(rd, "predictions.npz"))
    train_ids = set(pred["train_ids"].tolist())
    ids = np.asarray(full["node_id"]).astype(str)
    tr_idx = np.nonzero(np.fromiter((i in train_ids for i in ids), bool, count=len(ids)))[0]
    emb_all = hyb.embeddings(to_data(full, dev))
    y_all = full["y"]
    q = emb_all[full["labeled"]]
    out = []
    original = hyb.clf
    for f in KNN_TRAIN_FRACTIONS:
        sub = tr_idx if f >= 1.0 else rng.choice(tr_idx, size=int(round(f * len(tr_idx))), replace=False)
        hyb.fit_classifier(emb_all[sub], y_all[sub])
        t = _timeit(lambda: hyb.predict_proba_emb(q), "cpu", warmup, runs)
        t.update({"train_fraction": f, "n_stored": int(len(sub)), "n_queries": int(len(q))})
        out.append(t)
    hyb.clf = original
    return out


def write_summary():
    rows = []
    for d in C.DATASETS:
        dd = os.path.join(C.DIRS["timing"], d)
        if not os.path.isdir(dd):
            continue
        for fn in sorted(os.listdir(dd)):
            r = read_json(os.path.join(dd, fn))
            if not r:
                continue
            for t in r["results"]:
                rows.append({"dataset": d, "model": r["model"], "seed": r["seed"], "device": t["device"],
                             "fraction": t["fraction"], "n_nodes": t["n_nodes"], "n_edges": t["n_edges"],
                             "latency_mean_ms": 1e3 * t["mean_s"], "latency_sd_ms": 1e3 * t["sd_s"],
                             "cv_pct": t["cv_pct"], "throughput_nodes_per_s": t["nodes_per_s"],
                             "peak_gpu_gb": t["peak_gpu_mb"] / 1024 if t["peak_gpu_mb"] is not None else None,
                             "peak_process_rss_gb": t["peak_process_rss_gb"],
                             "median_ms": 1e3 * t["median_s"], "q1_ms": 1e3 * t["q1_s"], "q3_ms": 1e3 * t["q3_s"],
                             "n_params": r.get("n_params")})
    if rows:
        p = os.path.join(C.DIRS["timing"], "summary.csv")
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        LOG.info(f"Wrote {p}")


def main():
    global LOG
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", nargs="*", choices=C.DATASETS, default=C.DATASETS)
    ap.add_argument("--model", nargs="*", choices=TIMED_MODELS, default=TIMED_MODELS)
    ap.add_argument("--cpu-threads", type=int, default=8)
    ap.add_argument("--no-cpu", action="store_true")
    ap.add_argument("--force", action="store_true")
    args, _ = ap.parse_known_args()
    C.ensure_dirs()
    LOG = get_logger("timing", C.DIRS["logs"])
    import torch
    from jdata import DatasetBundle
    torch.set_num_threads(args.cpu_threads)
    devices = (["cuda"] if torch.cuda.is_available() else []) + ([] if args.no_cpu else ["cpu"])
    write_json(os.path.join(C.DIRS["timing"], "environment.json"),
               {**environment_info(), "cpu_threads": args.cpu_threads, "devices": devices})
    for d in args.dataset:
        bundle = DatasetBundle(d)
        for m in args.model:
            for s in TIMING_SEEDS:
                out = os.path.join(C.DIRS["timing"], d, f"{m.replace('+', '_plus_')}_seed{s}.json")
                if os.path.exists(out) and not args.force:
                    continue
                r = time_model(d, m, s, bundle, devices, C.TIMING_WARMUP, C.TIMING_RUNS)
                if r is not None:
                    write_json(out, r)
    write_summary()


if __name__ == "__main__":
    main()
