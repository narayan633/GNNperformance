import argparse
import os
import pickle

import numpy as np

import jconfig as C
from jutils import file_sha256, get_logger, now, write_json

LOG = None
EPS = 1e-8
F_NB_DEG, F_NB_AMT, F_NB_FRAUD = 31, 32, 33
SPLITS = ("train", "val", "test")


# ---------------------------------------------------------------------
# Neighbourhood step of TBNS
# ---------------------------------------------------------------------

def temporal_neighbors(G, node, ts_max, K):
    best = {}
    for pred in G.predecessors(node):
        ed = G.get_edge_data(pred, node)
        if not ed:
            continue
        t = max((float(a.get("timestamp", 0)) for a in ed.values()
                 if float(a.get("timestamp", 0)) <= ts_max), default=None)
        if t is not None and (pred not in best or t > best[pred]):
            best[pred] = t
    for succ in G.successors(node):
        ed = G.get_edge_data(node, succ)
        if not ed:
            continue
        t = max((float(a.get("timestamp", 0)) for a in ed.values()
                 if float(a.get("timestamp", 0)) <= ts_max), default=None)
        if t is not None and (succ not in best or t > best[succ]):
            best[succ] = t
    return [nd for nd, _ in sorted(best.items(), key=lambda x: x[1], reverse=True)[:K]]


def build_tbns_subgraph(G, seed_nodes, ts_max, K):
    all_nodes = set(seed_nodes)
    for nd in seed_nodes:
        all_nodes.update(temporal_neighbors(G, nd, ts_max, K))
    sub = G.subgraph(all_nodes).copy()
    prune = [(u, v, k) for u, v, k, a in sub.edges(keys=True, data=True)
             if float(a.get("timestamp", 0)) > ts_max]
    sub.remove_edges_from(prune)
    return sub, len(prune)


def neighbourhood_features(subgraph, node, is_train):

    f = np.zeros(3, dtype=np.float32)
    nbrs = list(set(list(subgraph.predecessors(node)) + list(subgraph.successors(node))))
    if nbrs:
        f[0] = float(np.mean([subgraph.degree(nb) for nb in nbrs]))
        nbr_a = [float(a.get("amount", 0.0)) for nb in nbrs for _, _, a in subgraph.edges(nb, data=True)]
        f[1] = float(np.mean(nbr_a)) if nbr_a else 0.0
        if is_train:
            f[2] = sum(1 for nb in nbrs if subgraph.nodes[nb].get("isp", 0) == 1) / (len(nbrs) + EPS)
    f[~np.isfinite(f)] = 0.0
    return f


# ---------------------------------------------------------------------

def main():
    global LOG
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", nargs="*", type=int, default=C.TBNS_K_VARIANTS)
    ap.add_argument("--force", action="store_true")
    args, _ = ap.parse_known_args()
    C.ensure_dirs()
    LOG = get_logger("tbns_build_k", C.DIRS["logs"])

    todo = [k for k in args.k if args.force or not all(os.path.exists(C.ethereum_pkl_for_k(k)[s]) for s in SPLITS)]
    if not todo:
        LOG.info("All K variants already built (use --force to rebuild)")
        return

    orig = {}
    for s in SPLITS:
        with open(C.ETHEREUM_PKL[s], "rb") as f:
            orig[s] = pickle.load(f)
    names = list(orig["train"]["feature_names"])
    assert names[F_NB_DEG] == "neighbor_avg_degree" and names[F_NB_AMT] == "neighbor_avg_amount" \
        and names[F_NB_FRAUD] == "neighbor_fraud_ratio", f"unexpected feature order: {names[30:]}"

    LOG.info(f"Loading {C.ETHEREUM_MULTIDIGRAPH} (a few minutes) ...")
    with open(C.ETHEREUM_MULTIDIGRAPH, "rb") as f:
        G = pickle.load(f)
    LOG.info(f"  {G.number_of_nodes():,} nodes | {G.number_of_edges():,} edges")
    graph_sha = file_sha256(C.ETHEREUM_MULTIDIGRAPH)

    check = {}
    for s in SPLITS:
        d = orig[s]
        seeds = list(d["node_ids"])
        sub, _ = build_tbns_subgraph(G, seeds, d["ts_max"], C.TBNS_K_MAIN)
        o = d["subgraph"]
        same_nodes = set(sub.nodes()) == set(o.nodes())
        same_edges = sub.number_of_edges() == o.number_of_edges()
        nb = np.stack([neighbourhood_features(sub, n, s == "train") for n in seeds])
        ref = np.asarray(d["X_raw"], dtype=np.float32)[:, F_NB_DEG:F_NB_FRAUD + 1]
        max_diff = float(np.max(np.abs(nb - ref) / np.maximum(np.abs(ref), 1.0)))
        check[s] = {"same_nodes": same_nodes, "same_edge_count": same_edges,
                    "nodes_rebuilt": sub.number_of_nodes(), "nodes_original": o.number_of_nodes(),
                    "edges_rebuilt": sub.number_of_edges(), "edges_original": o.number_of_edges(),
                    "max_rel_diff_features_31_33": max_diff}
        LOG.info(f"  K=3 check [{s}]: {check[s]}")
        if not (same_nodes and same_edges and max_diff < 1e-4):
            write_json(os.path.join(C.DIRS["tbns"], "k3_reproduction_check.json"), check)


    for k in todo:
        paths = C.ethereum_pkl_for_k(k)
        os.makedirs(os.path.dirname(paths["train"]), exist_ok=True)
        report = {"k": k, "built": now(), "multidigraph": C.ETHEREUM_MULTIDIGRAPH, "multidigraph_sha256": graph_sha,
                  "source_k3_sha256": {s: file_sha256(C.ETHEREUM_PKL[s]) for s in SPLITS}, "splits": {}}
        for s in SPLITS:
            d = orig[s]
            seeds = list(d["node_ids"])
            sub, n_pruned = build_tbns_subgraph(G, seeds, d["ts_max"], k)
            X = np.asarray(d["X_raw"], dtype=np.float32).copy()
            X[:, F_NB_DEG:F_NB_FRAUD + 1] = np.stack([neighbourhood_features(sub, n, s == "train") for n in seeds])
            out = dict(d)
            out.pop("X_norm", None)          # not used; would be stale for the recomputed columns
            out.update({"subgraph": sub, "X_raw": X, "tbns_k": k,
                        "stats": {**d.get("stats", {}), "subgraph_nodes": sub.number_of_nodes(),
                                  "subgraph_edges": sub.number_of_edges(), "tbns_k": k}})
            tmp = paths[s] + ".tmp"
            with open(tmp, "wb") as f:
                pickle.dump(out, f, protocol=4)
            os.replace(tmp, paths[s])
            report["splits"][s] = {"n_seeds": len(seeds), "subgraph_nodes": sub.number_of_nodes(),
                                   "subgraph_edges": sub.number_of_edges(), "pruned_non_causal": n_pruned,
                                   "k3_nodes": d["subgraph"].number_of_nodes(),
                                   "k3_edges": d["subgraph"].number_of_edges()}
            LOG.info(f"  K={k} [{s}] {report['splits'][s]}")
        write_json(os.path.join(os.path.dirname(paths["train"]), "build_report.json"), report)
    LOG.info("Done. Next: python data_build.py --dataset " + " ".join(f"ethereum_k{k}" for k in todo))


if __name__ == "__main__":
    main()
