Elliptic
    Rebuilt from the three released CSV files:
      - 165 features (CSV columns 2-166), directed edges as released
      - temporal split: train 1-34, val 35-42, test 43-49
      - z-score fitted on the training period (all nodes), applied to all
      - full graph with the released features for the random condition

Ethereum
    Loaded from the TBNS split graphs (eth_train/val/test.pkl, K = 3):
      - neighbor_fraud_ratio dropped (it uses neighbour labels in train only)
        pair (count and last timestamp kept), self-loops removed
      - z-score fitted on training seed nodes; neighbour-only nodes have
        no features in the released files and are set to 0 (= training mean)
      - monthly snapshots for EvolveGCN-H; each labelled node is evaluated
        once, in the month of its last transaction inside its split
      - a union graph of the three splits for the random condition

--------------------------------------------------------------------------------------

import argparse
import os
import pickle
import numpy as np
import jconfig as C
from jutils import write_json, write_npz, file_sha256, now, get_logger

LOG = None


# Common helpers

def save_graph(path, g):
    """g: dict with x, edge_index, y, labeled, node_id, time, has_features (+ optional extras)."""
    arrays = {}
    for k, v in g.items():
        if isinstance(v, np.ndarray):
            arrays[k] = v
    write_npz(path, **arrays)


def unique_directed_edges(src, dst, ts=None):
    src = np.asarray(src, dtype=np.int64)
    dst = np.asarray(dst, dtype=np.int64)
    keep = src != dst
    src, dst = src[keep], dst[keep]
    if ts is not None:
        ts = np.asarray(ts, dtype=np.float64)[keep]
    if src.size == 0:
        return np.zeros((2, 0), np.int64), np.zeros(0, np.int64), np.zeros(0, np.float64)
    n = int(max(src.max(), dst.max())) + 1
    key = src * n + dst
    order = np.argsort(key, kind="stable")
    key_s = key[order]
    uniq, start, counts = np.unique(key_s, return_index=True, return_counts=True)
    ei = np.stack([uniq // n, uniq % n]).astype(np.int64)
    if ts is None:
        last = np.zeros(len(uniq), np.float64)
    else:
        ts_s = ts[order]
        last = np.maximum.reduceat(ts_s, start)
    return ei, counts.astype(np.int64), last


def induced_subgraph(edge_index, node_mask):
    node_ids = np.nonzero(node_mask)[0]
    remap = -np.ones(node_mask.shape[0], dtype=np.int64)
    remap[node_ids] = np.arange(node_ids.shape[0])
    src, dst = edge_index
    keep = node_mask[src] & node_mask[dst]
    return np.stack([remap[src[keep]], remap[dst[keep]]]), node_ids


def graph_stats(g):
    lab = g["labeled"]
    y = g["y"]
    return {"n_nodes": int(g["x"].shape[0]), "n_edges": int(g["edge_index"].shape[1]),
            "n_labelled": int(lab.sum()), "n_fraud": int((y[lab] == 1).sum()),
            "fraud_rate_pct": float(100 * (y[lab] == 1).mean()) if lab.any() else 0.0,
            "n_features": int(g["x"].shape[1])}


# Elliptic
def build_elliptic(force=False):
    import pandas as pd
    out = C.DIRS["data_elliptic"]
    meta_path = os.path.join(out, "metadata.json")
    if os.path.exists(meta_path) and not force:
        LOG.info("Elliptic already built, skipping (use --force to rebuild)")
        return
    os.makedirs(out, exist_ok=True)
    LOG.info("Loading Elliptic CSV files ...")
    nf = C.ELLIPTIC_N_FEATURES
    feat_cols = ["txId", "time_step"] + [f"feat_{i}" for i in range(1, nf + 1)]
    features = pd.read_csv(C.ELLIPTIC_FEATURES_CSV, header=None, names=feat_cols)
    classes = pd.read_csv(C.ELLIPTIC_CLASSES_CSV)
    classes.columns = [c.strip() for c in classes.columns]
    classes = classes.rename(columns={c: "class" for c in classes.columns if c.strip().lower() == "class"})
    edges = pd.read_csv(C.ELLIPTIC_EDGELIST_CSV)
    edges.columns = [c.strip() for c in edges.columns]

    classes["class"] = classes["class"].astype(str).str.strip().map({"1": 1, "2": 0, "unknown": -1})
    table = features.merge(classes, on="txId", how="left")
    table["class"] = table["class"].fillna(-1).astype(int)
    table = table.reset_index(drop=True)
    assert table["time_step"].min() == 1 and table["time_step"].max() == C.ELLIPTIC_TOTAL_STEPS

    txid_to_idx = dict(zip(table["txId"].values, np.arange(len(table))))
    src = edges.iloc[:, 0].map(txid_to_idx)
    dst = edges.iloc[:, 1].map(txid_to_idx)
    valid = src.notna() & dst.notna()
    ei_full = np.stack([src[valid].astype(np.int64).values, dst[valid].astype(np.int64).values])
    n_dropped_edges = int((~valid).sum())

    fcols = [c for c in table.columns if c.startswith("feat_")]
    X = table[fcols].values.astype(np.float32)
    y = table["class"].values.astype(np.int64)
    ts = table["time_step"].values.astype(np.int64)
    tx = table["txId"].values.astype(np.int64)
    labeled = y != -1

    masks = {"train": (ts >= 1) & (ts <= C.ELLIPTIC_TRAIN_END),
             "val": (ts > C.ELLIPTIC_TRAIN_END) & (ts <= C.ELLIPTIC_VAL_END),
             "test": ts > C.ELLIPTIC_VAL_END}
    split_of = np.full(len(ts), -1)
    for i, s in enumerate(["train", "val", "test"]):
        split_of[masks[s]] = i
    n_cross = int((split_of[ei_full[0]] != split_of[ei_full[1]]).sum())
    LOG.info(f"Nodes {len(ts)}, edges {ei_full.shape[1]}, labelled {int(labeled.sum())}, cross-split edges {n_cross}")

    # z-score fitted on the training period (all nodes)
    mean = X[masks["train"]].mean(axis=0)
    std = X[masks["train"]].std(axis=0)
    std[std == 0] = 1.0
    Xn = ((X - mean) / std).astype(np.float32)

    meta = {"dataset": "elliptic", "built": now(), "n_features": nf, "feature_names": fcols,
            "split": {"train": [1, C.ELLIPTIC_TRAIN_END], "val": [C.ELLIPTIC_TRAIN_END + 1, C.ELLIPTIC_VAL_END],
                      "test": [C.ELLIPTIC_VAL_END + 1, C.ELLIPTIC_TOTAL_STEPS]},
            "normalisation": "z-score fitted on training-period nodes; random condition uses the released features",
            "n_dropped_edges": n_dropped_edges, "n_cross_split_edges": n_cross,
            "source_sha256": {p: file_sha256(p) for p in
                              (C.ELLIPTIC_FEATURES_CSV, C.ELLIPTIC_CLASSES_CSV, C.ELLIPTIC_EDGELIST_CSV)},
            "splits": {}, "snapshots": {}}

    # flat induced subgraph per split + per-step snapshots
    for s in ["train", "val", "test"]:
        ei, ids = induced_subgraph(ei_full, masks[s])
        g = {"x": Xn[ids], "edge_index": ei, "y": y[ids], "labeled": labeled[ids], "node_id": tx[ids],
             "time": ts[ids], "has_features": np.ones(len(ids), bool)}
        save_graph(os.path.join(out, f"temporal_{s}.npz"), g)
        meta["splits"][s] = graph_stats(g)
        steps = sorted(set(ts[masks[s]].tolist()))
        snap_dir = os.path.join(out, f"temporal_{s}_snapshots")
        os.makedirs(snap_dir, exist_ok=True)
        for t in steps:
            m = masks[s] & (ts == t)
            eis, idss = induced_subgraph(ei_full, m)
            gs = {"x": Xn[idss], "edge_index": eis, "y": y[idss], "labeled": labeled[idss], "node_id": tx[idss],
                  "time": ts[idss], "has_features": np.ones(len(idss), bool)}
            save_graph(os.path.join(snap_dir, f"t{t:03d}.npz"), gs)
        meta["snapshots"][s] = steps
        LOG.info(f"  {s}: {meta['splits'][s]}")

    # full graph (raw features) and full-period snapshots for the random condition
    g_full = {"x": X, "edge_index": ei_full, "y": y, "labeled": labeled, "node_id": tx, "time": ts,
              "has_features": np.ones(len(ts), bool)}
    save_graph(os.path.join(out, "full_graph.npz"), g_full)
    meta["full_graph"] = graph_stats(g_full)
    snap_dir = os.path.join(out, "full_snapshots")
    os.makedirs(snap_dir, exist_ok=True)
    for t in range(1, C.ELLIPTIC_TOTAL_STEPS + 1):
        m = ts == t
        eis, idss = induced_subgraph(ei_full, m)
        gs = {"x": X[idss], "edge_index": eis, "y": y[idss], "labeled": labeled[idss], "node_id": tx[idss],
              "time": ts[idss], "has_features": np.ones(len(idss), bool)}
        save_graph(os.path.join(snap_dir, f"t{t:03d}.npz"), gs)
    meta["snapshots"]["full"] = list(range(1, C.ELLIPTIC_TOTAL_STEPS + 1))
    write_npz(os.path.join(out, "normalisation.npz"), mean=mean, std=std)
    write_json(meta_path, meta)
    LOG.info("Elliptic done.")


# Ethereum (TBNS split graphs)
def month_of(ts):
    b = np.asarray(C.MONTH_BOUNDARIES, dtype=np.float64)
    return np.clip(np.searchsorted(b, np.asarray(ts, dtype=np.float64), side="right"), 1, C.ETHEREUM_N_MONTHS)


def load_tbns_split(path, drop_idx):
    with open(path, "rb") as f:
        d = pickle.load(f)
    sub = d["subgraph"]
    seeds = [str(n) for n in d["node_ids"]]
    X_raw = np.delete(np.asarray(d["X_raw"], dtype=np.float64), drop_idx, axis=1)
    y = np.asarray(d["y"], dtype=np.int64)
    multi = []   # (u, v, key, ts)
    for u, v, k, a in sub.edges(keys=True, data=True):
        multi.append((str(u), str(v), k, float(a.get("timestamp", 0.0))))
    nodes = [str(n) for n in sub.nodes()]
    return {"seeds": seeds, "X_raw": X_raw, "y": y, "multi": multi, "nodes": nodes,
            "ts_min": int(d["ts_min"]), "ts_max": int(d["ts_max"]),
            "feature_names": list(d["feature_names"]), "stats": d.get("stats", {})}

def _node_order(seeds, nodes):
    seed_set = set(seeds)
    others = sorted(n for n in nodes if n not in seed_set)
    order = list(seeds) + others
    return order, {n: i for i, n in enumerate(order)}


def _seed_eval_month(seeds, multi, month_lo, month_hi):
    last = {}
    if multi:
        months = month_of([t for _, _, _, t in multi])
        for (u, v, _, t), m in zip(multi, months.tolist()):
            if month_lo <= m <= month_hi:
                for n in (u, v):
                    if t > last.get(n, (-1.0, -1))[0]:
                        last[n] = (t, m)
    return np.array([last.get(s, (0.0, month_hi))[1] for s in seeds], dtype=np.int64)


def _monthly_snapshots(order, idx, x, y_all, labeled_all, has_feat, multi, seed_month, n_seeds, months):
    src = np.array([idx[u] for u, _, _, _ in multi], dtype=np.int64)
    dst = np.array([idx[v] for _, v, _, _ in multi], dtype=np.int64)
    mon = month_of([t for _, _, _, t in multi]) if multi else np.zeros(0, np.int64)
    snaps = []
    node_id_arr = np.array(order, dtype=object)
    for m in months:
        sel = mon == m
        ei_m, cnt, _ = unique_directed_edges(src[sel], dst[sel])
        active = set(ei_m.ravel().tolist())
        seeds_m = set(np.nonzero(seed_month == m)[0].tolist())   # seed indices are 0..n_seeds-1
        nodes_m = np.array(sorted(active | seeds_m), dtype=np.int64)
        if nodes_m.size == 0:
            continue
        remap = {g: i for i, g in enumerate(nodes_m.tolist())}
        if ei_m.shape[1]:
            ei_local = np.stack([np.array([remap[a] for a in ei_m[0]]), np.array([remap[b] for b in ei_m[1]])])
        else:
            ei_local = np.zeros((2, 0), np.int64)
        lab = np.array([(g < n_seeds) and (seed_month[g] == m) for g in nodes_m.tolist()], dtype=bool)
        snaps.append({"month": m, "x": x[nodes_m].astype(np.float32), "edge_index": ei_local.astype(np.int64),
                      "y": y_all[nodes_m], "labeled": lab & labeled_all[nodes_m],
                      "node_id": node_id_arr[nodes_m].astype(str), "time": np.full(len(nodes_m), m, np.int64),
                      "has_features": has_feat[nodes_m]})
    return snaps


def build_ethereum(force=False, name="ethereum", k=None):
    k = C.TBNS_K_MAIN if k is None else k
    pkl = C.ethereum_pkl_for_k(k)
    if not all(os.path.exists(p) for p in pkl.values()):
        LOG.warning(f"{name}: input files missing ({pkl['train']}). For K variants run tbns_build_k.py first.")
        return
    out = C.DIRS[f"data_{name}"]
    meta_path = os.path.join(out, "metadata.json")
    if os.path.exists(meta_path) and not force:
        LOG.info(f"{name} already built, skipping (use --force to rebuild)")
        return
    os.makedirs(out, exist_ok=True)

    with open(pkl["train"], "rb") as f:
        names_all = list(pickle.load(f)["feature_names"])
    drop_idx = [names_all.index(n) for n in C.ETHEREUM_DROP_FEATURES if n in names_all]
    feat_names = [n for i, n in enumerate(names_all) if i not in drop_idx]
    LOG.info(f"Ethereum: {len(names_all)} features in file, dropping {C.ETHEREUM_DROP_FEATURES}, "
             f"{len(feat_names)} kept")

    splits = {}
    for s in ["train", "val", "test"]:
        LOG.info(f"Loading {pkl[s]} ...")
        splits[s] = load_tbns_split(pkl[s], drop_idx)

    tr = splits["train"]["X_raw"]
    mean = tr.mean(axis=0)
    std = tr.std(axis=0)
    std[std < 1e-8] = 1.0

    month_range = {"train": (1, 33), "val": (34, 35), "test": (36, 42)}
    meta = {"dataset": name, "tbns_k": k, "built": now(), "n_features": len(feat_names), "feature_names": feat_names,
            "dropped_features": C.ETHEREUM_DROP_FEATURES, "month_range": month_range,
            "normalisation": "z-score fitted on training seed nodes; neighbour-only nodes set to 0",
            "edges": "parallel transactions collapsed to one directed edge per pair, self-loops removed",
            "source_sha256": {s: file_sha256(pkl[s]) for s in pkl},
            "splits": {}, "snapshots": {}}

    # temporal condition: one graph per split + monthly snapshots
    for s in ["train", "val", "test"]:
        d = splits[s]
        order, idx = _node_order(d["seeds"], d["nodes"])
        n_seeds = len(d["seeds"])
        N = len(order)
        x = np.zeros((N, len(feat_names)), np.float32)
        x[:n_seeds] = ((d["X_raw"] - mean) / std).astype(np.float32)
        has_feat = np.zeros(N, bool)
        has_feat[:n_seeds] = True
        y_all = np.full(N, -1, np.int64)
        y_all[:n_seeds] = d["y"]
        labeled = np.zeros(N, bool)
        labeled[:n_seeds] = True
        src = [idx[u] for u, _, _, _ in d["multi"]]
        dst = [idx[v] for _, v, _, _ in d["multi"]]
        tts = [t for _, _, _, t in d["multi"]]
        ei, cnt, last = unique_directed_edges(src, dst, tts)
        lo, hi = month_range[s]
        seed_month = _seed_eval_month(d["seeds"], d["multi"], lo, hi)
        time_all = np.zeros(N, np.int64)
        time_all[:n_seeds] = seed_month
        g = {"x": x, "edge_index": ei, "y": y_all, "labeled": labeled, "node_id": np.array(order, dtype=str),
             "time": time_all, "has_features": has_feat, "edge_count": cnt, "edge_last_ts": last}
        save_graph(os.path.join(out, f"temporal_{s}.npz"), g)
        st = graph_stats(g)
        st.update({"n_multi_edges": len(d["multi"]), "n_neighbour_only": N - n_seeds,
                   "n_edges_after_split_end": int(sum(t > d["ts_max"] for t in tts))})
        meta["splits"][s] = st
        snaps = _monthly_snapshots(order, idx, x, y_all, labeled, has_feat, d["multi"], seed_month, n_seeds,
                                   list(range(lo, hi + 1)))
        snap_dir = os.path.join(out, f"temporal_{s}_snapshots")
        os.makedirs(snap_dir, exist_ok=True)
        for sn in snaps:
            m = sn.pop("month")
            save_graph(os.path.join(snap_dir, f"t{m:03d}.npz"), sn)
        meta["snapshots"][s] = [int(sn_m) for sn_m in range(lo, hi + 1)
                                if os.path.exists(os.path.join(snap_dir, f"t{sn_m:03d}.npz"))]
        lab_in_snaps = sum(int(np.load(os.path.join(snap_dir, f"t{m:03d}.npz"))["labeled"].sum())
                           for m in meta["snapshots"][s])
        assert lab_in_snaps == n_seeds, f"{s}: {lab_in_snaps} labelled in snapshots vs {n_seeds} seeds"
        LOG.info(f"  {s}: {st}")

    # overlap report
    seed_sets = {s: set(splits[s]["seeds"]) for s in splits}
    meta["seed_overlap"] = {"train_val": len(seed_sets["train"] & seed_sets["val"]),
                            "train_test": len(seed_sets["train"] & seed_sets["test"]),
                            "val_test": len(seed_sets["val"] & seed_sets["test"])}

    # random condition: union graph, raw features, first occurrence of a seed wins
    seeds_u, X_u, y_u, origin, month_u = [], [], [], [], []
    seen = set()
    for s in ["train", "val", "test"]:
        d = splits[s]
        lo, hi = month_range[s]
        sm = _seed_eval_month(d["seeds"], d["multi"], lo, hi)
        for i, n in enumerate(d["seeds"]):
            if n in seen:
                continue
            seen.add(n)
            seeds_u.append(n)
            X_u.append(d["X_raw"][i])
            y_u.append(int(d["y"][i]))
            origin.append(s)
            month_u.append(int(sm[i]))
    all_nodes = set()
    multi_u = {}
    for s in ["train", "val", "test"]:
        all_nodes.update(splits[s]["nodes"])
        for u, v, k, t in splits[s]["multi"]:
            multi_u[(u, v, k)] = t
    order, idx = _node_order(seeds_u, all_nodes)
    n_seeds = len(seeds_u)
    N = len(order)
    x = np.zeros((N, len(feat_names)), np.float32)
    x[:n_seeds] = np.asarray(X_u, dtype=np.float32)
    has_feat = np.zeros(N, bool)
    has_feat[:n_seeds] = True
    y_all = np.full(N, -1, np.int64)
    y_all[:n_seeds] = np.asarray(y_u)
    labeled = np.zeros(N, bool)
    labeled[:n_seeds] = True
    multi_list = [(u, v, k, t) for (u, v, k), t in multi_u.items()]
    src = [idx[u] for u, _, _, _ in multi_list]
    dst = [idx[v] for _, v, _, _ in multi_list]
    ei, cnt, last = unique_directed_edges(src, dst, [t for *_, t in multi_list])
    time_all = np.zeros(N, np.int64)
    time_all[:n_seeds] = np.asarray(month_u)
    origin_arr = np.array(origin + [""] * (N - n_seeds), dtype=str)
    g = {"x": x, "edge_index": ei, "y": y_all, "labeled": labeled, "node_id": np.array(order, dtype=str),
         "time": time_all, "has_features": has_feat, "edge_count": cnt, "edge_last_ts": last,
         "origin_split": origin_arr}
    save_graph(os.path.join(out, "full_graph.npz"), g)
    meta["full_graph"] = graph_stats(g)
    meta["full_graph"]["n_multi_edges"] = len(multi_list)
    snaps = _monthly_snapshots(order, idx, x, y_all, labeled, has_feat, multi_list, time_all[:n_seeds], n_seeds,
                               list(range(1, C.ETHEREUM_N_MONTHS + 1)))
    snap_dir = os.path.join(out, "full_snapshots")
    os.makedirs(snap_dir, exist_ok=True)
    months_saved = []
    for sn in snaps:
        m = sn.pop("month")
        months_saved.append(int(m))
        save_graph(os.path.join(snap_dir, f"t{m:03d}.npz"), sn)
    meta["snapshots"]["full"] = months_saved
    write_npz(os.path.join(out, "normalisation.npz"), mean=mean, std=std)
    write_json(meta_path, meta)
    LOG.info(f"  random union graph: {meta['full_graph']}")
    LOG.info(f"{name} done.")


def main():
    global LOG
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", nargs="*", choices=C.DATASETS + C.K_DATASETS, default=None,
                    help="default: elliptic, ethereum and any K variant whose files exist")
    ap.add_argument("--force", action="store_true")
    args, _ = ap.parse_known_args()
    C.ensure_dirs()
    LOG = get_logger("data_build", C.DIRS["logs"])
    todo = args.dataset or (C.DATASETS + [d for d, k in zip(C.K_DATASETS, C.TBNS_K_VARIANTS)
                                          if os.path.exists(C.ethereum_pkl_for_k(k)["train"])])
    if "elliptic" in todo:
        build_elliptic(args.force)
    if "ethereum" in todo:
        build_ethereum(args.force)
    for d, k in zip(C.K_DATASETS, C.TBNS_K_VARIANTS):
        if d in todo:
            build_ethereum(args.force, name=d, k=k)


if __name__ == "__main__":
    main()
