import argparse
import os

import numpy as np

import jconfig as C
from jutils import release_gpu, RunLock, get_logger, is_done, mark_done, now, read_json, write_json

LOG = None
DATASET = "ethereum"
VARIANTS = ["structure_only", "rewired"]
MODELS = C.MAIN_MODELS          # all seven models


def _month_end_ts(month):
    return C.MONTH_BOUNDARIES[month]          # boundary after month m (index m)


def audit(name="ethereum"):
    base = C.DIRS[f"data_{name}"]
    meta = read_json(os.path.join(base, "metadata.json"))
    if meta is None:
        LOG.info(f"{name} not built, audit skipped")
        return None
    out = {"dataset": name, "tbns_k": meta.get("tbns_k", C.TBNS_K_MAIN), "built": now(), "metadata_splits": meta["splits"], "seed_overlap": meta["seed_overlap"],
           "full_graph": meta["full_graph"], "splits": {}}
    node_sets = {}
    for s in ("train", "val", "test"):
        g = np.load(os.path.join(base, f"temporal_{s}.npz"))
        lo, hi = meta["month_range"][s]
        ei = g["edge_index"]
        n = g["x"].shape[0]
        lab = g["labeled"]
        last_ts = g["edge_last_ts"]
        end_ts = _month_end_ts(hi)
        start_ts = C.MONTH_BOUNDARIES[lo - 1]
        # label reach: labelled neighbours (undirected) of each labelled seed
        und = np.concatenate([ei, ei[::-1]], axis=1)
        lab_nb = np.zeros(n, np.int64)
        np.add.at(lab_nb, und[0], lab[und[1]].astype(np.int64))
        deg = np.bincount(und[0], minlength=n)
        node_sets[s] = set(np.asarray(g["node_id"]).astype(str).tolist())
        out["splits"][s] = {
            "months": [lo, hi], "n_nodes": int(n), "n_edges": int(ei.shape[1]),
            "n_labelled": int(lab.sum()), "fraud_rate_pct": float(100 * g["y"][lab].mean()),
            "neighbour_only_share": float(1 - lab.mean()),
            "density": float(ei.shape[1] / max(n * (n - 1), 1)),
            "mean_parallel_tx_per_edge": float(g["edge_count"].mean()) if len(g["edge_count"]) else 0.0,
            "edges_after_window_end": int((last_ts >= end_ts).sum()),
            "edges_before_window_start": int((last_ts < start_ts).sum()),
            "edges_after_window_end_note": "last transaction timestamp of a collapsed edge compared with split end",
            "seed_degree_median": float(np.median(deg[lab])), "seed_degree_p95": float(np.percentile(deg[lab], 95)),
            "isolated_seed_share": float((deg[lab] == 0).mean()),
            "seeds_with_labelled_neighbour_share": float((lab_nb[lab] > 0).mean()),
            "fraud_seeds_with_fraud_neighbour_share": float(
                np.mean([(g["y"][und[1][und[0] == i]] == 1).any() for i in np.nonzero(lab & (g["y"] == 1))[0]]))
            if (lab & (g["y"] == 1)).any() else None,
        }
    out["node_sharing"] = {f"{a}_{b}": len(node_sets[a] & node_sets[b]) for a, b in
                           (("train", "val"), ("train", "test"), ("val", "test"))}
    out["causal_check_passed"] = all(out["splits"][s]["edges_after_window_end"] == 0 for s in out["splits"])
    fn = "audit.json" if name == "ethereum" else f"audit_{name}.json"
    write_json(os.path.join(C.DIRS["tbns"], fn), out)
    LOG.info(f"{name} audit written; causal check passed = {out['causal_check_passed']}")
    return out


# ---------------------------------------------------------------------
# Part B
# ---------------------------------------------------------------------

def transform_graph(g, variant, rng):
   
    g2 = dict(g)
    ei = g["edge_index"]
    n = g["x"].shape[0]
    if variant == "structure_only":
        din = np.bincount(ei[1], minlength=n)
        dout = np.bincount(ei[0], minlength=n)
        g2["x"] = np.stack([np.ones(n), np.log1p(din), np.log1p(dout)], 1).astype(np.float32)
        g2["has_features"] = np.ones(n, bool)
    elif variant == "rewired":
        if ei.shape[1] == 0:
            return g2
        dst = rng.permutation(ei[1])
        keep = ei[0] != dst
        pairs = np.unique(np.stack([ei[0][keep], dst[keep]], 1), axis=0)
        g2["edge_index"] = pairs.T.astype(np.int64).reshape(2, -1)
    else:
        raise ValueError(variant)
    return g2


class VariantBundle:


    def __init__(self, bundle, variant, rng_seed):
        self.dataset = bundle.dataset
        self.variant = variant
        self._base = bundle
        self._seed = rng_seed
        rng = np.random.default_rng(rng_seed)
        self.flat = {s: transform_graph(g, variant, rng) for s, g in bundle.flat.items()}
        self.full = None
        self._snaps = {}
        self._nf = int(self.flat["train"]["x"].shape[1])

    def snapshots(self, which):
        if which == "full":
            raise ValueError("structure tests use the temporal condition only")
        if not self._snaps:
            rng = np.random.default_rng([self._seed, 1])
            for s in ("train", "val", "test"):
                self._snaps[s] = [transform_graph(g, self.variant, rng) for g in self._base.snapshots(s)]
        return self._snaps[which]

    @property
    def n_features(self):
        return self._nf


def train_variants(device):
    from params import get_best_params
    from jdata import DatasetBundle
    from jtrain import run_one
    bundle = DatasetBundle(DATASET)
    for variant in VARIANTS:
        for m in MODELS:
            params = get_best_params(m, DATASET, "temporal")
            for s in C.SEEDS_10:
                rd = os.path.join(C.DIRS["tbns"], "runs", variant, m.replace("+", "_plus_"), f"seed{s}")
                if is_done(rd):
                    continue
                with RunLock(rd) as lk:
                    if not lk.acquired or is_done(rd):
                        continue
                    vb = VariantBundle(bundle, variant, rng_seed=s)   # rewiring differs per seed
                    try:
                        r = run_one(m, vb, "temporal", s, params, device, run_dir=rd)
                    except Exception as e:
                        LOG.exception(f"{variant} {m} seed{s} failed: {e}")
                        continue
                    if m in C.SEQUENCE_MODELS:
                        n_edges = {k: int(sum(g["edge_index"].shape[1] for g in vb.snapshots(k)))
                                   for k in ("train", "val", "test")}
                    else:
                        n_edges = {k: int(v["edge_index"].shape[1]) for k, v in vb.flat.items()}
                    r.update({"variant": variant, "n_edges": n_edges,
                              "graph_input": "monthly snapshots" if m in C.SEQUENCE_MODELS else "flat split graph"})
                    write_json(os.path.join(rd, "metrics.json"), r)
                    mark_done(rd)
                    release_gpu(device)
                    LOG.info(f"{variant} {m} seed{s} test AUC-PRC {r['metrics']['test']['auc_prc']:.4f}")


def main():
    global LOG
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit-only", action="store_true")
    ap.add_argument("--device", default="cuda")
    args, _ = ap.parse_known_args()
    C.ensure_dirs()
    LOG = get_logger("tbns", C.DIRS["logs"])
    for name in ["ethereum"] + C.K_DATASETS:
        audit(name)
    if args.audit_only:
        return
    import torch
    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    train_variants(device)


if __name__ == "__main__":
    main()
