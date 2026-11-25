"""
explanation_stability.py
==============================================================
"""

import argparse
import csv
import os
import time

import numpy as np

import jconfig as C
from jutils import release_gpu, RunLock, get_logger, now, read_json, run_dir_for, set_seed, write_json, write_npz

LOG = None
E = C.EXPLAIN
NODE_BUDGET = int(os.environ.get("CRYPTOEX_EXPLAIN_NODE_BUDGET", 500_000))
EDGE_BUDGET = 4 * NODE_BUDGET
GRAD_MODELS = {"GCN", "GAT", "GCN+LSTM", "GNN+Autoencoder"}
EXPLAIN_CONDITIONS = ["temporal", "random70"]
SHAP_CONVERGENCE = [500, 1000, 2000]


# =====================================================================
# Node selection
# =====================================================================

def select_nodes(inp, condition, dataset):
    path = os.path.join(C.DIRS["explain"], dataset, condition, "nodes.json")
    sel = read_json(path)
    if sel is not None:
        return sel
    rng = np.random.default_rng(12345)
    out = {}
    for role, split, n in (("test", "test", E["n_test_nodes"]), ("train", "train", E["n_train_nodes"])):
        g = inp["raw"][split]
        m = inp["masks"][split].cpu().numpy() & g["has_features"]
        y = g["y"]
        ids = np.asarray(g["node_id"]).astype(str)
        n_f = int(round(n * E["fraud_share"]))
        fr = np.nonzero(m & (y == 1))[0]
        li = np.nonzero(m & (y == 0))[0]
        pick = np.concatenate([rng.choice(fr, min(n_f, len(fr)), replace=False),
                               rng.choice(li, min(n - n_f, len(li)), replace=False)])
        out[role] = sorted(ids[pick].tolist())
    write_json(path, {**out, "rule": "fixed rng 12345, half fraud, from seed-42 split if random", "time": now()})
    return read_json(path)


# =====================================================================
# Batched model function on a target nodes subgraph
# =====================================================================

class NodeFunction:

    def __init__(self, kind, obj, data, target_global, device):
        import torch
        from torch_geometric.utils import k_hop_subgraph
        self.kind, self.obj, self.device = kind, obj, device
        subset, ei, mapping, _ = k_hop_subgraph(int(target_global), E["num_hops"], data.edge_index,
                                                relabel_nodes=True, num_nodes=data.num_nodes)
        self.x = data.x[subset].detach()
        self.ei = ei
        self.t = int(mapping[0])
        self.n = int(subset.numel())
        self.e = int(ei.size(1))
        self.x_target = self.x[self.t].cpu().numpy().astype(np.float64)
        self.torch = torch

    def _forward(self, xb, b):
        torch = self.torch
        n, e = self.n, self.e
        x = self.x.repeat(b, 1)
        idx = torch.arange(b, device=self.device) * n + self.t
        x[idx] = xb
        off = (torch.arange(b, device=self.device) * n).repeat_interleave(e)
        ei = self.ei.repeat(1, b) + off.unsqueeze(0)
        if self.kind == "hybrid":
            emb = self.obj.gcn.get_embeddings(x, ei)[idx]
            return emb
        return torch.exp(self.obj(x, ei))[idx, 1]

    def __call__(self, X):
        torch = self.torch
        X = np.atleast_2d(np.asarray(X, dtype=np.float32))
        chunk = max(1, min(NODE_BUDGET // max(self.n, 1), EDGE_BUDGET // max(self.e, 1)))
        out = []
        with torch.no_grad():
            for i in range(0, len(X), chunk):
                xb = torch.as_tensor(X[i:i + chunk], device=self.device)
                r = self._forward(xb, xb.size(0))
                if self.kind == "hybrid":
                    out.append(self.obj.predict_proba_emb(r.cpu().numpy()))
                else:
                    out.append(r.cpu().numpy())
        return np.concatenate(out).astype(np.float64)

    def proba2(self, X):
        p = self(X)
        return np.stack([1 - p, p], 1)

    def differentiable(self, xt):
        """Gradient path for the feature-mask explainer (single copy)."""
        x = self.x.clone()
        x[self.t] = xt
        return self.torch.exp(self.obj(x, self.ei))[self.t, 1]


# =====================================================================
# Explainers
# =====================================================================

def explain_shap(f, background, nsamples, seed):
    import shap
    np.random.seed(seed)
    ex = shap.KernelExplainer(f, background, link="identity")
    v = ex.shap_values(f.x_target.reshape(1, -1), nsamples=nsamples, l1_reg=False, silent=True)
    v = np.asarray(v[1] if isinstance(v, list) else v).reshape(-1)
    return v


LIME_VERSION = 2   


def explain_lime(f, lime_train, nsamples, seed):
    """LIME with local sampling: perturbations are drawn around the explained node
    (x_target + N(0, 1) * training sd), so the surrogate is fitted locally to that node.
    The lime default (sample_around_instance=False) samples around the training mean,
    which makes explanations nearly identical across nodes."""
    from lime.lime_tabular import LimeTabularExplainer
    ex = LimeTabularExplainer(lime_train, mode="classification", discretize_continuous=False,
                              sample_around_instance=True, random_state=seed, feature_selection="none")
    F = lime_train.shape[1]
    r = ex.explain_instance(f.x_target, f.proba2, labels=(1,), num_features=F, num_samples=nsamples)
    v = np.zeros(F)
    for j, w in r.as_map()[1]:
        v[j] = w
    return v


def explain_mask(f, baseline, epochs, seed):
    torch = f.torch
    torch.manual_seed(seed)
    xt = torch.as_tensor(f.x_target, dtype=torch.float32, device=f.device)
    base = torch.as_tensor(baseline, dtype=torch.float32, device=f.device)
    with torch.no_grad():
        p0 = float(f.differentiable(xt))
    target_cls = 1 if p0 >= 0.5 else 0
    w = torch.nn.Parameter(0.1 * torch.randn(xt.numel(), device=f.device))
    opt = torch.optim.Adam([w], lr=0.01)
    for p in f.obj.parameters():
        p.requires_grad_(False)
    for _ in range(epochs):
        opt.zero_grad()
        m = torch.sigmoid(w)
        p = f.differentiable(m * xt + (1 - m) * base)
        pc = p if target_cls == 1 else 1 - p
        ent = -(m * torch.log(m + 1e-15) + (1 - m) * torch.log(1 - m + 1e-15))
        loss = -torch.log(pc + 1e-15) + 1.0 * m.mean() + 0.1 * ent.mean()
        loss.backward()
        opt.step()
    return torch.sigmoid(w).detach().cpu().numpy().astype(np.float64)


# =====================================================================
# One (dataset, condition, model, seed)
# =====================================================================

def explain_run(dataset, condition, model, seed, bundle, device):
    import shap
    import torch
    from jtrain import load_model, prepare_flat
    rd = run_dir_for(C.DIRS["runs"], "E2", dataset, condition, model, seed)
    if not os.path.exists(os.path.join(rd, "DONE")):
        LOG.warning(f"missing {rd}")
        return None
    set_seed(seed)
    inp = prepare_flat(bundle, condition, seed, device)
    sel_inp = inp if seed == C.SEEDS_EXPLAIN[0] or condition == "temporal" else \
        prepare_flat(bundle, condition, C.SEEDS_EXPLAIN[0], device)
    sel = select_nodes(sel_inp, condition, dataset)
    kind, obj = load_model(rd, device)

    # background and LIME reference data: training nodes with features
    g_tr = inp["raw"]["train"]
    x_tr = inp["data"]["train"].x.cpu().numpy()
    mtr = inp["masks"]["train"].cpu().numpy() & g_tr["has_features"]
    rng = np.random.default_rng(seed)
    pool = x_tr[mtr]
    lime_train = pool[rng.choice(len(pool), min(2000, len(pool)), replace=False)].astype(np.float64)
    background = shap.kmeans(lime_train, E["shap_background"])  
    baseline = pool.mean(0)
    sd = pool.std(0)
    sd[sd < 1e-8] = 1.0

    # full-graph probabilities for the exactness check
    full_p = {}
    for split in ("train", "test"):
        d = inp["data"][split]
        with torch.no_grad():
            if kind == "hybrid":
                full_p[split] = obj.predict_proba_emb(obj.embeddings(d))
            else:
                full_p[split] = torch.exp(obj(d.x, d.edge_index))[:, 1].cpu().numpy()

    rec = {k: [] for k in ("node_id", "role", "y", "prob_full", "prob_sub", "sub_nodes", "sub_edges",
                           "shap", "lime", "mask", "t_shap", "t_lime", "t_mask")}
    noise = {k: [] for k in ("node_id", "shap", "lime", "mask", "prob_noisy")}
    conv = {k: [] for k in ("node_id", "shap_n", "shap_rep", "shap", "lime_rep", "lime")}
    for role in ("test", "train"):
        d = inp["data"][role]
        ids = np.asarray(inp["raw"][role]["node_id"]).astype(str)
        pos = {n: i for i, n in enumerate(ids)}
        for k, nid in enumerate(sel[role]):
            gi = pos[nid]
            f = NodeFunction(kind, obj, d, gi, device)
            p_sub = float(f(f.x_target.reshape(1, -1))[0])
            t0 = time.perf_counter()
            a_shap = explain_shap(f, background, E["shap_nsamples"], seed)
            t1 = time.perf_counter()
            a_lime = explain_lime(f, lime_train, E["lime_nsamples"], seed)
            t2 = time.perf_counter()
            a_mask = explain_mask(f, baseline, E["gnnexplainer_epochs"], seed) if model in GRAD_MODELS else \
                np.full(len(a_shap), np.nan)
            t3 = time.perf_counter()
            for key, v in (("node_id", nid), ("role", role), ("y", int(inp["raw"][role]["y"][gi])),
                           ("prob_full", float(full_p[role][gi])), ("prob_sub", p_sub), ("sub_nodes", f.n),
                           ("sub_edges", f.e), ("shap", a_shap), ("lime", a_lime), ("mask", a_mask),
                           ("t_shap", t1 - t0), ("t_lime", t2 - t1), ("t_mask", (t3 - t2) if model in GRAD_MODELS else np.nan)):
                rec[key].append(v)

            if role == "test" and k < E["noise_nodes"]:
                nrng = np.random.default_rng([seed, k])
                xt0 = f.x_target.copy()
                f.x_target = xt0 + nrng.standard_normal(len(xt0)) * sd * E["noise_level"]
                f.x = f.x.clone()
                f.x[f.t] = torch.as_tensor(f.x_target, dtype=f.x.dtype, device=f.x.device)
                noise["node_id"].append(nid)
                noise["prob_noisy"].append(float(f(f.x_target.reshape(1, -1))[0]))
                noise["shap"].append(explain_shap(f, background, E["shap_nsamples"], seed))
                noise["lime"].append(explain_lime(f, lime_train, E["lime_nsamples"], seed))
                noise["mask"].append(explain_mask(f, baseline, E["gnnexplainer_epochs"], seed)
                                     if model in GRAD_MODELS else np.full(len(a_shap), np.nan))
                f.x_target = xt0
                f.x[f.t] = torch.as_tensor(xt0, dtype=f.x.dtype, device=f.x.device)

            if role == "test" and k < E["convergence_nodes"]:
                for ns in SHAP_CONVERGENCE:
                    for rep in range(2):
                        conv["node_id"].append(nid)
                        conv["shap_n"].append(ns)
                        conv["shap_rep"].append(rep)
                        conv["shap"].append(explain_shap(f, background, ns, seed + 1000 * (rep + 1)))
                        conv["lime_rep"].append(rep)
                        conv["lime"].append(explain_lime(f, lime_train, E["lime_nsamples"], seed + 1000 * (rep + 1))
                                            if ns == E["shap_nsamples"] else np.full(len(a_shap), np.nan))
    arr = {k: np.asarray(v) for k, v in rec.items()}
    arr.update({f"noise_{k}": np.asarray(v) for k, v in noise.items()})
    arr.update({f"conv_{k}": np.asarray(v) for k, v in conv.items()})
    meta = {"dataset": dataset, "condition": condition, "model": model, "seed": seed, "source_run": rd,
            "explain_config": E, "grad_explainer": model in GRAD_MODELS, "time": now(),
            "max_abs_sub_vs_full": float(np.max(np.abs(arr["prob_sub"] - arr["prob_full"]))),
            "mean_time_s": {k: float(np.nanmean(arr[f"t_{k}"])) for k in ("shap", "lime", "mask")}
            if model in GRAD_MODELS else {k: float(np.nanmean(arr[f"t_{k}"])) for k in ("shap", "lime")}}
    if meta["max_abs_sub_vs_full"] > 1e-3:
        LOG.warning(f"subgraph prob differs from full graph by {meta['max_abs_sub_vs_full']:.2e} ({model}, {dataset})")
    return arr, meta


# =====================================================================
# Stability summaries
# =====================================================================

def topk_jaccard(a, b, k):
    ta = set(np.argsort(-np.abs(a))[:k].tolist())
    tb = set(np.argsort(-np.abs(b))[:k].tolist())
    return len(ta & tb) / len(ta | tb)


def rank_corr(a, b):
    from scipy.stats import spearmanr
    if np.all(np.isnan(a)) or np.all(np.isnan(b)):
        return np.nan
    return float(spearmanr(np.abs(a), np.abs(b)).correlation)


def _load(dataset, condition, model, seed):
    p = os.path.join(C.DIRS["explain"], dataset, condition, model.replace("+", "_plus_"), f"seed{seed}.npz")
    return dict(np.load(p, allow_pickle=False)) if os.path.exists(p) else None


def summarise():
    k = E["top_k"]
    rows_seed, rows_expl, rows_time, rows_noise, rows_floor, rows_rt = [], [], [], [], [], []
    for d in C.DATASETS:
        for c in EXPLAIN_CONDITIONS:
            for m in C.EXPLAIN_MODELS:
                runs = {s: _load(d, c, m, s) for s in C.SEEDS_EXPLAIN}
                runs = {s: r for s, r in runs.items() if r is not None}
                if not runs:
                    continue
                expls = ["shap", "lime"] + (["mask"] if m in GRAD_MODELS else []) + \
                    (["gnnx"] if all("gnnx" in r for r in runs.values()) else [])
                base = {"dataset": d, "condition": c, "model": m}
                # across seeds
                ss = sorted(runs)
                for ex in expls:
                    for i in range(len(ss)):
                        for j in range(i + 1, len(ss)):
                            A, B = runs[ss[i]], runs[ss[j]]
                            ib = {n: t for t, n in enumerate(B["node_id"])}
                            for t, n in enumerate(A["node_id"]):
                                if n in ib:
                                    rows_seed.append({**base, "explainer": ex, "seed_a": ss[i], "seed_b": ss[j],
                                                      "node_id": n, "role": A["role"][t], "y": int(A["y"][t]),
                                                      "topk_jaccard": topk_jaccard(A[ex][t], B[ex][ib[n]], k),
                                                      "spearman": rank_corr(A[ex][t], B[ex][ib[n]])})
                for s, r in runs.items():
                    # across explainers
                    for a_i in range(len(expls)):
                        for b_i in range(a_i + 1, len(expls)):
                            ea, eb = expls[a_i], expls[b_i]
                            for t in range(len(r["node_id"])):
                                rows_expl.append({**base, "seed": s, "pair": f"{ea}-{eb}", "node_id": r["node_id"][t],
                                                  "role": r["role"][t],
                                                  "topk_jaccard": topk_jaccard(r[ea][t], r[eb][t], k),
                                                  "spearman": rank_corr(r[ea][t], r[eb][t])})
                    # across time (global importance)
                    for ex in expls:
                        tr = np.nanmean(np.abs(r[ex][r["role"] == "train"]), 0)
                        te = np.nanmean(np.abs(r[ex][r["role"] == "test"]), 0)
                        rows_time.append({**base, "seed": s, "explainer": ex, "topk_jaccard": topk_jaccard(tr, te, k),
                                          "top10_jaccard": topk_jaccard(tr, te, 10), "spearman": rank_corr(tr, te)})
                    # under noise
                    pos = {n: t for t, n in enumerate(r["node_id"]) if r["role"][t] == "test"}
                    for q, n in enumerate(r["noise_node_id"]):
                        t = pos[n]
                        for ex in expls:
                            rows_noise.append({**base, "seed": s, "explainer": ex, "node_id": n,
                                               "prob_change": float(r["noise_prob_noisy"][q] - r["prob_sub"][t]),
                                               "topk_jaccard": topk_jaccard(r[ex][t], r[f"noise_{ex}"][q], k),
                                               "spearman": rank_corr(r[ex][t], r[f"noise_{ex}"][q])})
                    # explainer noise floor (repeat runs)
                    cn = r["conv_node_id"]
                    for n in np.unique(cn):
                        for ns in SHAP_CONVERGENCE:
                            sel = np.nonzero((cn == n) & (r["conv_shap_n"] == ns))[0]
                            if len(sel) == 2:
                                rows_floor.append({**base, "seed": s, "explainer": "shap", "nsamples": int(ns),
                                                   "node_id": n,
                                                   "topk_jaccard": topk_jaccard(r["conv_shap"][sel[0]], r["conv_shap"][sel[1]], k),
                                                   "spearman": rank_corr(r["conv_shap"][sel[0]], r["conv_shap"][sel[1]])})
                                if ns == E["shap_nsamples"]:
                                    rows_floor.append({**base, "seed": s, "explainer": "lime",
                                                       "nsamples": E["lime_nsamples"], "node_id": n,
                                                       "topk_jaccard": topk_jaccard(r["conv_lime"][sel[0]], r["conv_lime"][sel[1]], k),
                                                       "spearman": rank_corr(r["conv_lime"][sel[0]], r["conv_lime"][sel[1]])})
                                    if "gnnx" in expls:
                                        rows_floor.append({**base, "seed": s, "explainer": "gnnx",
                                                           "nsamples": E["gnnexplainer_epochs"], "node_id": n,
                                                           "topk_jaccard": topk_jaccard(r["conv_gnnx"][sel[0]], r["conv_gnnx"][sel[1]], k),
                                                           "spearman": rank_corr(r["conv_gnnx"][sel[0]], r["conv_gnnx"][sel[1]])})
                    for ex in expls:
                        tt = r[f"t_{ex}"]
                        rows_rt.append({**base, "seed": s, "explainer": ex, "mean_s": float(np.nanmean(tt)),
                                        "median_s": float(np.nanmedian(tt)), "max_s": float(np.nanmax(tt)),
                                        "mean_sub_nodes": float(np.mean(r["sub_nodes"])),
                                        "max_abs_sub_vs_full": float(np.max(np.abs(r["prob_sub"] - r["prob_full"])))})
    rows_edges, rows_cover = edge_stability_rows()
    for name, rows in (("seeds", rows_seed), ("explainers", rows_expl), ("time", rows_time), ("noise", rows_noise),
                       ("explainer_floor", rows_floor), ("runtime", rows_rt), ("gnnx_edges", rows_edges),
                       ("gnnx_edge_coverage", rows_cover)):
        if not rows:
            continue
        p = os.path.join(C.DIRS["explain"], f"summary_{name}.csv")
        with open(p, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        LOG.info(f"Wrote {p} ({len(rows)} rows)")


def lime_rerun(dataset, condition, model, seed, bundle, device):
    import torch
    from jtrain import load_model, prepare_flat
    od = os.path.join(C.DIRS["explain"], dataset, condition, model.replace("+", "_plus_"))
    npz = os.path.join(od, f"seed{seed}.npz")
    meta_p = os.path.join(od, f"seed{seed}.json")
    meta = read_json(meta_p) or {}
    if not os.path.exists(npz) or meta.get("lime_version") == LIME_VERSION:
        return "skip"
    old = dict(np.load(npz, allow_pickle=False))
    rd = run_dir_for(C.DIRS["runs"], "E2", dataset, condition, model, seed)
    set_seed(seed)
    inp = prepare_flat(bundle, condition, seed, device)
    sel = read_json(os.path.join(C.DIRS["explain"], dataset, condition, "nodes.json"))
    kind, obj = load_model(rd, device)
    # identical reference data and noise scale as explain_run
    g_tr = inp["raw"]["train"]
    x_tr = inp["data"]["train"].x.cpu().numpy()
    mtr = inp["masks"]["train"].cpu().numpy() & g_tr["has_features"]
    rng = np.random.default_rng(seed)
    pool = x_tr[mtr]
    lime_train = pool[rng.choice(len(pool), min(2000, len(pool)), replace=False)].astype(np.float64)
    sd = pool.std(0)
    sd[sd < 1e-8] = 1.0

    lime, t_lime, noise_lime, conv_lime, order = [], [], [], [], []
    for role in ("test", "train"):
        d = inp["data"][role]
        ids = np.asarray(inp["raw"][role]["node_id"]).astype(str)
        pos = {n: i for i, n in enumerate(ids)}
        for k, nid in enumerate(sel[role]):
            f = NodeFunction(kind, obj, d, pos[nid], device)
            t0 = time.perf_counter()
            lime.append(explain_lime(f, lime_train, E["lime_nsamples"], seed))
            t_lime.append(time.perf_counter() - t0)
            order.append(nid)
            if role == "test" and k < E["noise_nodes"]:
                nrng = np.random.default_rng([seed, k])
                xt0 = f.x_target.copy()
                f.x_target = xt0 + nrng.standard_normal(len(xt0)) * sd * E["noise_level"]
                f.x = f.x.clone()
                f.x[f.t] = torch.as_tensor(f.x_target, dtype=f.x.dtype, device=f.x.device)
                noise_lime.append(explain_lime(f, lime_train, E["lime_nsamples"], seed))
                f.x_target = xt0
                f.x[f.t] = torch.as_tensor(xt0, dtype=f.x.dtype, device=f.x.device)
            if role == "test" and k < E["convergence_nodes"]:
                for ns in SHAP_CONVERGENCE:
                    for rep in range(2):
                        conv_lime.append(explain_lime(f, lime_train, E["lime_nsamples"], seed + 1000 * (rep + 1))
                                         if ns == E["shap_nsamples"] else np.full(len(f.x_target), np.nan))
    assert list(old["node_id"]) == order, "node order differs from the saved run"
    assert len(noise_lime) == len(old["noise_node_id"]) and len(conv_lime) == len(old["conv_node_id"])
    new = dict(old)
    for key, val in (("lime", lime), ("t_lime", t_lime), ("noise_lime", noise_lime), ("conv_lime", conv_lime)):
        new[f"{key}_v1"] = old[key]
        new[key] = np.asarray(val)
    write_npz(npz, **new)
    meta.update({"lime_version": LIME_VERSION, "lime_sampling": "around the explained node (sample_around_instance=True)",
                 "lime_rerun_time": now(), "lime_mean_time_s_v2": float(np.mean(t_lime))})
    if "mean_time_s" in meta:
        meta["mean_time_s"]["lime_v1"] = meta["mean_time_s"].get("lime")
        meta["mean_time_s"]["lime"] = float(np.mean(t_lime))
    write_json(meta_p, meta)
    return "ok"


# =====================================================================
# GNNExplainer 
# =====================================================================
GNNX_VERSION = 1
EDGE_TOP_K = 5


def _gnnx_explainer(model, with_edges):
    from torch_geometric.explain import Explainer, GNNExplainer
    return Explainer(model=model, algorithm=GNNExplainer(epochs=E["gnnexplainer_epochs"]),
                     explanation_type="model", node_mask_type="common_attributes",
                     edge_mask_type="object" if with_edges else None,
                     model_config=dict(mode="multiclass_classification", task_level="node", return_type="log_probs"))


def explain_gnnx(f, obj, seed, explainers):
    """Original GNNExplainer: one feature mask shared by all nodes of the subgraph (as in the paper)
    plus one mask weight per edge, learned jointly for the prediction of the target node.
    Returns (feature mask [F], edge mask [E], edge sources, edge targets) on the subgraph."""
    torch = f.torch
    torch.manual_seed(seed)
    ex = explainers[f.e > 0]
    for p in obj.parameters():
        p.requires_grad_(False)
    out = ex(f.x, f.ei, index=int(f.t))
    feat = out.node_mask.detach().reshape(-1).cpu().numpy().astype(np.float64)
    assert feat.shape[0] == f.x.shape[1], f"feature mask has {feat.shape[0]} values, expected {f.x.shape[1]}"
    edge = out.edge_mask.detach().cpu().numpy().astype(np.float64) if f.e > 0 else np.zeros(0)
    return feat, edge


def gnnx_run(dataset, condition, model, seed, bundle, device, max_nodes=None, write=True):
    """Adds GNNExplainer results to a finished run (same nodes, noise draws and repeat seeds as
    SHAP, LIME and the mask). Existing arrays are not touched."""
    import torch
    from torch_geometric.utils import k_hop_subgraph
    from jtrain import load_model, prepare_flat
    od = os.path.join(C.DIRS["explain"], dataset, condition, model.replace("+", "_plus_"))
    npz = os.path.join(od, f"seed{seed}.npz")
    meta_p = os.path.join(od, f"seed{seed}.json")
    meta = read_json(meta_p) or {}
    if model not in GRAD_MODELS or not os.path.exists(npz) or (write and meta.get("gnnx_version") == GNNX_VERSION):
        return "skip"
    old = dict(np.load(npz, allow_pickle=False))
    rd = run_dir_for(C.DIRS["runs"], "E2", dataset, condition, model, seed)
    set_seed(seed)
    inp = prepare_flat(bundle, condition, seed, device)
    sel = read_json(os.path.join(C.DIRS["explain"], dataset, condition, "nodes.json"))
    kind, obj = load_model(rd, device)
    obj.eval()
    explainers = {True: _gnnx_explainer(obj, True), False: _gnnx_explainer(obj, False)}
    g_tr = inp["raw"]["train"]
    x_tr = inp["data"]["train"].x.cpu().numpy()
    mtr = inp["masks"]["train"].cpu().numpy() & g_tr["has_features"]
    sd = x_tr[mtr].std(0)
    sd[sd < 1e-8] = 1.0

    feat, t_g, order = [], [], []
    noise_feat, conv_feat = [], []
    edges = {"main": ([], [], [], []), "noise": ([], [], [], []), "conv": ([], [], [], [])}   # ptr-less lists of arrays

    def keep_edges(bucket, f, d, edge):
        ids = np.asarray(d.node_id).astype(str)
        subset, _, _, _ = k_hop_subgraph(int(f_global[0]), E["num_hops"], d.edge_index, relabel_nodes=True,
                                         num_nodes=d.num_nodes)
        sub_ids = ids[subset.cpu().numpy()]
        has = f.e > 0 and len(edge) == f.e          # an empty edge array (skipped repeat) stores no edges
        src = sub_ids[f.ei[0].cpu().numpy()] if has else np.zeros(0, dtype=str)
        dst = sub_ids[f.ei[1].cpu().numpy()] if has else np.zeros(0, dtype=str)
        b = edges[bucket]
        b[0].append(len(edge))
        b[1].append(src)
        b[2].append(dst)
        b[3].append(edge)

    n_done = 0
    for role in ("test", "train"):
        d = inp["data"][role]
        ids = np.asarray(inp["raw"][role]["node_id"]).astype(str)
        pos = {n: i for i, n in enumerate(ids)}
        for k, nid in enumerate(sel[role]):
            if max_nodes is not None and n_done >= max_nodes:
                break
            n_done += 1
            f_global = [pos[nid]]
            f = NodeFunction(kind, obj, d, pos[nid], device)
            t0 = time.perf_counter()
            fm, em = explain_gnnx(f, obj, seed, explainers)
            t_g.append(time.perf_counter() - t0)
            feat.append(fm)
            order.append(nid)
            keep_edges("main", f, d, em)
            if role == "test" and k < E["noise_nodes"]:
                nrng = np.random.default_rng([seed, k])
                xt0 = f.x_target.copy()
                f.x = f.x.clone()
                f.x[f.t] = torch.as_tensor(xt0 + nrng.standard_normal(len(xt0)) * sd * E["noise_level"],
                                           dtype=f.x.dtype, device=f.x.device)
                fm_n, em_n = explain_gnnx(f, obj, seed, explainers)
                noise_feat.append(fm_n)
                keep_edges("noise", f, d, em_n)
                f.x[f.t] = torch.as_tensor(xt0, dtype=f.x.dtype, device=f.x.device)
            if role == "test" and k < E["convergence_nodes"]:
                for ns in SHAP_CONVERGENCE:
                    for rep in range(2):
                        if ns == E["shap_nsamples"]:
                            fm_c, em_c = explain_gnnx(f, obj, seed + 1000 * (rep + 1), explainers)
                        else:
                            fm_c, em_c = np.full(len(f.x_target), np.nan), np.zeros(0)
                        conv_feat.append(fm_c)
                        keep_edges("conv", f, d, em_c)
    if not write:
        return {"n": len(feat), "mean_s": float(np.mean(t_g)), "feat_len": len(feat[0]),
                "edges_first": int(edges["main"][0][0])}
    assert list(old["node_id"]) == order, "node order differs from the saved run"
    assert len(noise_feat) == len(old["noise_node_id"]) and len(conv_feat) == len(old["conv_node_id"])
    new = dict(old)
    new.update({"gnnx": np.asarray(feat), "t_gnnx": np.asarray(t_g), "noise_gnnx": np.asarray(noise_feat),
                "conv_gnnx": np.asarray(conv_feat)})
    for bucket, (lens, src, dst, val) in edges.items():
        new[f"gnnx_edge_{bucket}_ptr"] = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
        new[f"gnnx_edge_{bucket}_src"] = np.concatenate(src).astype(str) if src else np.zeros(0, dtype=str)
        new[f"gnnx_edge_{bucket}_dst"] = np.concatenate(dst).astype(str) if dst else np.zeros(0, dtype=str)
        new[f"gnnx_edge_{bucket}_val"] = np.concatenate(val) if val else np.zeros(0)
    write_npz(npz, **new)
    meta.update({"gnnx_version": GNNX_VERSION, "gnnx_epochs": E["gnnexplainer_epochs"],
                 "gnnx_setting": "PyG GNNExplainer, node_mask_type=common_attributes, edge_mask_type=object",
                 "gnnx_mean_time_s": float(np.mean(t_g)), "gnnx_time": now()})
    write_json(meta_p, meta)
    return "ok"


def _edge_sets(r, bucket):
    ptr = r[f"gnnx_edge_{bucket}_ptr"]
    out = []
    for i in range(len(ptr) - 1):
        a, b = ptr[i], ptr[i + 1]
        val = np.asarray(r[f"gnnx_edge_{bucket}_val"][a:b], dtype=np.float64)
        keys = np.asarray([f"{u}>{v}" for u, v in zip(r[f"gnnx_edge_{bucket}_src"][a:b], r[f"gnnx_edge_{bucket}_dst"][a:b])])
        act = val > 0
        top = set(keys[act][np.argsort(-val[act], kind="stable")[:EDGE_TOP_K]].tolist()) if act.any() else set()
        out.append((len(keys), int(act.sum()), top))
    return out


def _jac(a, b):
    return len(a & b) / len(a | b) if (a | b) else np.nan


_CHANCE = {}


def _pair_jac(g, n, k):
    a, b = set(g.choice(n, k, replace=False).tolist()), set(g.choice(n, k, replace=False).tolist())
    return len(a & b) / len(a | b)


def _chance_jaccard(n, k=EDGE_TOP_K, draws=2000):

    if n not in _CHANCE:
        g = np.random.default_rng(n)
        _CHANCE[n] = float(np.mean([_pair_jac(g, n, k) for _ in range(draws)]))
    return _CHANCE[n]


def _edge_row(base, view, sa, sb, n, ne, na, nb, ta, tb):
    m = min(na, nb)
    return {**base, "view": view, "seed_a": sa, "seed_b": sb, "node_id": n, "n_edges": ne, "n_active": m,
            "top5_edge_jaccard": _jac(ta, tb), "chance_jaccard": _chance_jaccard(m)}


def edge_stability_rows():

    rows, cover = [], []
    for d in C.DATASETS:
        for c in EXPLAIN_CONDITIONS:
            for m in C.EXPLAIN_MODELS:
                runs = {s: _load(d, c, m, s) for s in C.SEEDS_EXPLAIN}
                runs = {s: r for s, r in runs.items() if r is not None and "gnnx_edge_main_ptr" in r}
                if not runs:
                    continue
                base = {"dataset": d, "condition": c, "model": m}
                main = {s: dict(zip(r["node_id"], _edge_sets(r, "main"))) for s, r in runs.items()}
                for s in main:
                    for n, (ne, na, _) in main[s].items():
                        cover.append({**base, "seed": s, "node_id": n, "n_edges": ne, "n_active": na})
                ss = sorted(runs)
                for i in range(len(ss)):
                    for j in range(i + 1, len(ss)):
                        for n, (ne, na, ta) in main[ss[i]].items():
                            if n in main[ss[j]]:
                                _, nb, tb = main[ss[j]][n]
                                if na > EDGE_TOP_K and nb > EDGE_TOP_K:
                                    rows.append(_edge_row(base, "seeds", ss[i], ss[j], n, ne, na, nb, ta, tb))
                for s, r in runs.items():
                    for q, (ne, nb, tb) in enumerate(_edge_sets(r, "noise")):
                        n = r["noise_node_id"][q]
                        _, na, ta = main[s][n]
                        if na > EDGE_TOP_K and nb > EDGE_TOP_K:
                            rows.append(_edge_row(base, "noise", s, s, n, ne, na, nb, ta, tb))
                    conv = _edge_sets(r, "conv")
                    cn = r["conv_node_id"]
                    for n in np.unique(cn):
                        idx = np.nonzero((cn == n) & (r["conv_shap_n"] == E["shap_nsamples"]))[0]
                        if len(idx) == 2:
                            (ne, na, ta), (_, nb, tb) = conv[idx[0]], conv[idx[1]]
                            if na > EDGE_TOP_K and nb > EDGE_TOP_K:
                                rows.append(_edge_row(base, "repeat_floor", s, s, n, ne, na, nb, ta, tb))
    return rows, cover


def main():
    global LOG
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", nargs="*", choices=C.DATASETS, default=C.DATASETS)
    ap.add_argument("--model", nargs="*", choices=C.EXPLAIN_MODELS, default=C.EXPLAIN_MODELS)
    ap.add_argument("--condition", nargs="*", choices=EXPLAIN_CONDITIONS, default=EXPLAIN_CONDITIONS)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--summarise", action="store_true")
    ap.add_argument("--gnnx-only", action="store_true",
                    help="add GNNExplainer (feature + edge masks) to finished runs, then summarise")
    ap.add_argument("--gnnx-smoke", action="store_true",
                    help="explain 3 nodes of one run with GNNExplainer, print timing, write nothing")
    ap.add_argument("--lime-only", action="store_true",
                    help="recompute only LIME (local sampling) for finished runs, then summarise")
    args, _ = ap.parse_known_args()
    C.ensure_dirs()
    LOG = get_logger("explain", C.DIRS["logs"])
    if args.summarise:
        summarise()
        return
    import torch
    from jdata import DatasetBundle
    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    if args.gnnx_smoke:
        d, c, m, s = args.dataset[0], args.condition[0], [x for x in args.model if x in GRAD_MODELS][0], C.SEEDS_EXPLAIN[0]
        r = gnnx_run(d, c, m, s, DatasetBundle(d), device, max_nodes=3, write=False)
        LOG.info(f"GNNExplainer smoke {d} {c} {m} seed{s}: {r}")
        print("SMOKE OK" if isinstance(r, dict) else f"SMOKE NOT RUN: {r}")
        return
    if args.gnnx_only:
        for d in args.dataset:
            bundle = DatasetBundle(d)
            for c in args.condition:
                for m in [x for x in args.model if x in GRAD_MODELS]:
                    for s in C.SEEDS_EXPLAIN:
                        od = os.path.join(C.DIRS["explain"], d, c, m.replace("+", "_plus_"))
                        with RunLock(os.path.join(od, f"gnnxlock_seed{s}")) as lk:
                            if not lk.acquired:
                                continue
                            t0 = time.perf_counter()
                            try:
                                st = gnnx_run(d, c, m, s, bundle, device)
                            except Exception as e:
                                LOG.exception(f"GNNExplainer {d} {c} {m} seed{s} failed: {e}")
                                continue
                            if st == "ok":
                                LOG.info(f"GNNExplainer {d} {c} {m} seed{s} done in {time.perf_counter() - t0:.0f}s")
                            release_gpu(device)
        summarise()
        return
    if args.lime_only:
        for d in args.dataset:
            bundle = DatasetBundle(d)
            for c in args.condition:
                for m in args.model:
                    for s in C.SEEDS_EXPLAIN:
                        od = os.path.join(C.DIRS["explain"], d, c, m.replace("+", "_plus_"))
                        with RunLock(os.path.join(od, f"limelock_seed{s}")) as lk:
                            if not lk.acquired:
                                continue
                            t0 = time.perf_counter()
                            try:
                                st = lime_rerun(d, c, m, s, bundle, device)
                            except Exception as e:
                                LOG.exception(f"LIME rerun {d} {c} {m} seed{s} failed: {e}")
                                continue
                            if st == "ok":
                                LOG.info(f"LIME {d} {c} {m} seed{s} done in {time.perf_counter() - t0:.0f}s")
                            release_gpu(device)
        summarise()
        return
    for d in args.dataset:
        bundle = DatasetBundle(d)
        for c in args.condition:
            for m in args.model:
                for s in C.SEEDS_EXPLAIN:
                    od = os.path.join(C.DIRS["explain"], d, c, m.replace("+", "_plus_"))
                    out = os.path.join(od, f"seed{s}.npz")
                    if os.path.exists(out):
                        continue
                    with RunLock(os.path.join(od, f"lock_seed{s}")) as lk:
                        if not lk.acquired or os.path.exists(out):
                            continue
                        t0 = time.perf_counter()
                        try:
                            res = explain_run(d, c, m, s, bundle, device)
                        except Exception as e:
                            LOG.exception(f"{d} {c} {m} seed{s} failed: {e}")
                            continue
                        if res is None:
                            continue
                        arr, meta = res
                        meta["wall_s"] = time.perf_counter() - t0
                        write_json(os.path.join(od, f"seed{s}.json"), meta)
                        write_npz(out, **arr)
                        LOG.info(f"{d} {c} {m} seed{s} explained in {meta['wall_s']:.0f}s")
                        release_gpu(device)
    summarise()


if __name__ == "__main__":
    main()
