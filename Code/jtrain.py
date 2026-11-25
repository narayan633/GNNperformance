import os
import pickle
import time

import numpy as np
import torch
import torch.nn.functional as F

import jconfig as C
from jdata import masks_from_ids, node_degree, normalise_random, random_split, to_data
from jmetrics import best_f1_threshold, compute_metrics
from jmodels import HybridGCN, RawBaseline, ae_combined_loss, build_torch_model, clone_state
from sklearn.metrics import average_precision_score


def _auc_prc(y, p):
    y = np.asarray(y)
    return float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else 0.0


def _class_weights(y, device):
    n1 = max(int((y == 1).sum()), 1)
    n0 = int((y == 0).sum())
    return torch.tensor([1.0, n0 / n1], dtype=torch.float32, device=device)


def _sync(device):
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


# =====================================================================
# Building the inputs for one run
# =====================================================================

def prepare_flat(bundle, condition, seed, device):
    """Returns dict with graphs and masks for train/val/test, plus split info for saving."""
    if condition == "temporal":
        g = {s: bundle.flat[s] for s in ("train", "val", "test")}
        data = {s: to_data(g[s], device) for s in g}
        masks = {s: data[s].labeled_mask for s in g}
        return {"data": data, "masks": masks, "raw": g, "norm": None, "split_ids": None}
    full = bundle.full
    ids = random_split(full, seed, condition)
    m = masks_from_ids(full["node_id"], ids, full["labeled"])
    x = full["x"]
    norm = None
    if C.NORMALISE_RANDOM[bundle.dataset]:
        x, mu, sd = normalise_random(full["x"], m["train"], full["has_features"])
        norm = {"mu": mu, "sd": sd}
    g = dict(full)
    g["x"] = x
    d = to_data(g, device)
    data = {"train": d, "val": d, "test": d}
    masks = {s: torch.as_tensor(m[s], device=device) for s in m}
    return {"data": data, "masks": masks, "raw": {"train": g, "val": g, "test": g}, "norm": norm,
            "split_ids": ids}


def prepare_sequence(bundle, condition, seed, device):
    if condition == "temporal":
        seqs = {s: bundle.snapshots(s) for s in ("train", "val", "test")}
        data = {s: [to_data(g, device) for g in seqs[s]] for s in seqs}
        masks = {s: [d.labeled_mask for d in data[s]] for s in data}
        return {"data": data, "masks": masks, "raw": seqs, "norm": None, "split_ids": None, "mode": "temporal"}
    full = bundle.full
    ids = random_split(full, seed, condition)
    snaps = bundle.snapshots("full")
    norm = None
    if C.NORMALISE_RANDOM[bundle.dataset]:
        m_full = masks_from_ids(full["node_id"], ids, full["labeled"])
        _, mu, sd = normalise_random(full["x"], m_full["train"], full["has_features"])
        norm = {"mu": mu, "sd": sd}
    data, masks = [], {"train": [], "val": [], "test": []}
    raw = []
    for g in snaps:
        g2 = dict(g)
        if norm is not None:
            xn = ((g["x"] - norm["mu"]) / norm["sd"]).astype(np.float32)
            xn[~g["has_features"]] = 0.0
            g2["x"] = xn
        raw.append(g2)
        d = to_data(g2, device)
        data.append(d)
        mm = masks_from_ids(g["node_id"], ids, g["labeled"])
        for s in masks:
            masks[s].append(torch.as_tensor(mm[s], device=device))
    return {"data": {"full": data}, "masks": masks, "raw": {"full": raw}, "norm": norm, "split_ids": ids,
            "mode": "random"}


# =====================================================================
# Training loops
# =====================================================================

def _train_flat_torch(name, model, inp, params, epochs, device):
    opt = torch.optim.Adam(model.parameters(), lr=params["learning_rate"], weight_decay=params["weight_decay"])
    dtr, dva = inp["data"]["train"], inp["data"]["val"]
    mtr, mva = inp["masks"]["train"], inp["masks"]["val"]
    cw = _class_weights(dtr.y[mtr], device)
    alpha = 1.0 if name == "GNN+Autoencoder-noRecon" else params.get("alpha", 0.7)
    best, best_state, bad, best_epoch, ep = -1.0, None, 0, 0, 0
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        if name.startswith("GNN+Autoencoder"):
            loss = ae_combined_loss(model, dtr.x, dtr.edge_index, dtr.y, mtr, cw, alpha)
        else:
            out = model(dtr.x, dtr.edge_index)
            loss = F.nll_loss(out[mtr], dtr.y[mtr], weight=cw)
        loss.backward()
        opt.step()
        if ep % C.EVAL_EVERY == 0 or ep == epochs - 1:
            model.eval()
            with torch.no_grad():
                p = torch.exp(model(dva.x, dva.edge_index))[:, 1][mva].cpu().numpy()
            score = _auc_prc(dva.y[mva].cpu().numpy(), p)
            if score > best:
                best, best_state, bad, best_epoch = score, clone_state(model), 0, ep
            else:
                bad += 1
                if bad >= C.PATIENCE:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    return {"best_val_auc_prc_during_training": best, "best_epoch": best_epoch, "epochs_run": ep + 1}


def _predict_flat_torch(model, inp, split):
    d, m = inp["data"][split], inp["masks"][split]
    model.eval()
    with torch.no_grad():
        p = torch.exp(model(d.x, d.edge_index))[:, 1]
    return p[m].cpu().numpy(), m.cpu().numpy()


def _train_sequence(model, inp, params, epochs, device):
    opt = torch.optim.Adam(model.parameters(), lr=params["learning_rate"], weight_decay=params["weight_decay"])
    if inp["mode"] == "temporal":
        tr = inp["data"]["train"]
        trm = inp["masks"]["train"]
        y_all = torch.cat([d.y[m] for d, m in zip(tr, trm)])
    else:
        tr = inp["data"]["full"]
        trm = inp["masks"]["train"]
        y_all = torch.cat([d.y[m] for d, m in zip(tr, trm)])
    cw = _class_weights(y_all, device)
    best, best_state, bad, best_epoch, ep = -1.0, None, 0, 0, 0
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        outs = model.run_sequence(tr)
        o = torch.cat([out[m] for out, m in zip(outs, trm)])
        loss = F.nll_loss(o, y_all, weight=cw)
        loss.backward()
        opt.step()
        if ep % C.EVAL_EVERY == 0 or ep == epochs - 1:
            p, y = _predict_sequence(model, inp, "val")
            score = _auc_prc(y, p)
            if score > best:
                best, best_state, bad, best_epoch = score, clone_state(model), 0, ep
            else:
                bad += 1
                if bad >= C.PATIENCE:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    return {"best_val_auc_prc_during_training": best, "best_epoch": best_epoch, "epochs_run": ep + 1}


def _sequence_outputs(model, inp, split):

    model.eval()
    with torch.no_grad():
        if inp["mode"] == "temporal":
            order = ["train", "val", "test"]
            seq = []
            for s in order[:order.index(split) + 1]:
                seq += inp["data"][s]
            outs = model.run_sequence(seq)
            n = len(inp["data"][split])
            return list(zip(outs[-n:], inp["data"][split], inp["masks"][split]))
        outs = model.run_sequence(inp["data"]["full"])
        return list(zip(outs, inp["data"]["full"], inp["masks"][split]))


def _predict_sequence(model, inp, split):
    res = _sequence_outputs(model, inp, split)
    p = np.concatenate([torch.exp(o)[:, 1][m].cpu().numpy() for o, _, m in res]) if res else np.zeros(0)
    y = np.concatenate([d.y[m].cpu().numpy() for _, d, m in res]) if res else np.zeros(0)
    return p, y


def _train_hybrid(name, inp, params, epochs, device, seed):
    kind = "xgb" if name.startswith("GCN+XGBoost") else "knn"
    hyb = HybridGCN(inp["data"]["train"].num_features, params, kind, seed, no_edge=name in C.NOEDGE_MODELS)
    hyb.gcn.to(device)
    info = _train_flat_torch("GCN", hyb.gcn, inp, params, epochs, device)
    dtr, mtr = inp["data"]["train"], inp["masks"]["train"]
    emb_tr = hyb.embeddings(dtr)[mtr.cpu().numpy()]
    hyb.fit_classifier(emb_tr, dtr.y[mtr].cpu().numpy())
    return hyb, info


def _predict_hybrid(hyb, inp, split):
    d, m = inp["data"][split], inp["masks"][split]
    mm = m.cpu().numpy()
    return hyb.predict_proba_emb(hyb.embeddings(d)[mm]), mm


# =====================================================================
# Prediction records
# =====================================================================

def _flat_record(inp, split, p, mask_np):
    g = inp["raw"][split]
    deg = node_degree(g["edge_index"], g["x"].shape[0])
    return {"node_id": np.asarray(g["node_id"]).astype(str)[mask_np], "y": g["y"][mask_np].astype(np.int8),
            "prob": p.astype(np.float32), "time": g["time"][mask_np].astype(np.int32),
            "degree": deg[mask_np].astype(np.int32), "has_features": g["has_features"][mask_np]}


def _sequence_record(model, inp, split):
    res = _sequence_outputs(model, inp, split)
    raws = inp["raw"][split] if inp["mode"] == "temporal" else inp["raw"]["full"]
    parts = {k: [] for k in ("node_id", "y", "prob", "time", "degree", "has_features")}
    for (o, _d, m), g in zip(res, raws):
        mm = m.cpu().numpy()
        deg = node_degree(g["edge_index"], g["x"].shape[0])
        parts["node_id"].append(np.asarray(g["node_id"]).astype(str)[mm])
        parts["y"].append(g["y"][mm].astype(np.int8))
        parts["prob"].append(torch.exp(o)[:, 1].cpu().numpy()[mm].astype(np.float32))
        parts["time"].append(g["time"][mm].astype(np.int32))
        parts["degree"].append(deg[mm].astype(np.int32))
        parts["has_features"].append(g["has_features"][mm])
    return {k: np.concatenate(v) if v else np.zeros(0) for k, v in parts.items()}


# =====================================================================
# Main entry
# =====================================================================

def run_one(model_name, bundle, condition, seed, params, device, epochs=C.FINAL_EPOCHS, run_dir=None):

    from jutils import set_seed
    set_seed(seed)
    t0 = time.perf_counter()
    is_seq = model_name in C.SEQUENCE_MODELS
    inp = prepare_sequence(bundle, condition, seed, device) if is_seq else prepare_flat(bundle, condition, seed, device)
    prep_s = time.perf_counter() - t0

    _sync(device)
    t1 = time.perf_counter()
    model_obj = None
    if model_name in C.RAW_MODELS:
        kind = "xgb" if model_name == "XGBoost-raw" else "knn"
        rb = RawBaseline(params, kind, seed)
        g = inp["raw"]["train"]
        mtr = inp["masks"]["train"].cpu().numpy()
        rb.fit(g["x"][mtr], g["y"][mtr])
        info = {"best_epoch": None, "epochs_run": None}
        model_obj = rb
    elif model_name in C.HYBRID_MODELS:
        model_obj, info = _train_hybrid(model_name, inp, params, epochs, device, seed)
    elif is_seq:
        model_obj = build_torch_model(model_name, bundle.n_features, params).to(device)
        info = _train_sequence(model_obj, inp, params, epochs, device)
    else:
        model_obj = build_torch_model(model_name, bundle.n_features, params).to(device)
        info = _train_flat_torch(model_name, model_obj, inp, params, epochs, device)
    _sync(device)
    train_s = time.perf_counter() - t1

    # predictions on val and test
    recs = {}
    for split in ("val", "test"):
        if model_name in C.RAW_MODELS:
            g = inp["raw"][split]
            mm = inp["masks"][split].cpu().numpy()
            recs[split] = _flat_record(inp, split, model_obj.predict_proba(g["x"][mm]), mm)
        elif model_name in C.HYBRID_MODELS:
            p, mm = _predict_hybrid(model_obj, inp, split)
            recs[split] = _flat_record(inp, split, p, mm)
        elif is_seq:
            recs[split] = _sequence_record(model_obj, inp, split)
        else:
            p, mm = _predict_flat_torch(model_obj, inp, split)
            recs[split] = _flat_record(inp, split, p, mm)

    thr = best_f1_threshold(recs["val"]["y"], recs["val"]["prob"])
    metrics = {"val": compute_metrics(recs["val"]["y"], recs["val"]["prob"], thr, C.ECE_BINS),
               "test": compute_metrics(recs["test"]["y"], recs["test"]["prob"], thr, C.ECE_BINS)}
    result = {"model": model_name, "dataset": bundle.dataset, "condition": condition, "seed": seed,
              "params": params, "threshold": thr, "metrics": metrics, "train_info": info,
              "time_prepare_s": prep_s, "time_train_s": train_s,
              "n_train": int(inp["masks"]["train"].sum()) if not is_seq else int(sum(int(m.sum()) for m in inp["masks"]["train"])),
              }

    if run_dir is not None:
        os.makedirs(run_dir, exist_ok=True)
        arrays = {}
        for split, r in recs.items():
            for k, v in r.items():
                arrays[f"{split}_{k}"] = v
        if inp["split_ids"] is not None:
            arrays["train_ids"] = np.array(sorted(inp["split_ids"]["train"]), dtype=str)
        from jutils import write_npz
        write_npz(os.path.join(run_dir, "predictions.npz"), **arrays)
        if inp["norm"] is not None:
            write_npz(os.path.join(run_dir, "normalisation.npz"), **inp["norm"])
        save_model(model_name, model_obj, params, bundle.n_features, run_dir)
    return result


def save_model(model_name, model_obj, params, n_features, run_dir):
    meta = {"model": model_name, "params": params, "n_features": n_features}
    if model_name in C.RAW_MODELS:
        with open(os.path.join(run_dir, "classifier.pkl"), "wb") as f:
            pickle.dump(model_obj.clf, f)
    elif model_name in C.HYBRID_MODELS:
        torch.save({"state_dict": model_obj.gcn.state_dict(), **meta}, os.path.join(run_dir, "model.pt"))
        with open(os.path.join(run_dir, "classifier.pkl"), "wb") as f:
            pickle.dump(model_obj.clf, f)
    else:
        torch.save({"state_dict": model_obj.state_dict(), **meta}, os.path.join(run_dir, "model.pt"))


def load_model(run_dir, device="cpu"):

    from jmodels import GCN, GCNNoEdge
    pt = os.path.join(run_dir, "model.pt")
    clf_p = os.path.join(run_dir, "classifier.pkl")
    if os.path.exists(pt):
        ck = torch.load(pt, map_location=device, weights_only=False)
        name, p, nf = ck["model"], ck["params"], ck["n_features"]
        if name in C.HYBRID_MODELS:
            kind = "xgb" if name.startswith("GCN+XGBoost") else "knn"
            no_edge = name in C.NOEDGE_MODELS
            hyb = HybridGCN(nf, p, kind, seed=0, no_edge=no_edge)
            hyb.gcn = (GCNNoEdge if no_edge else GCN)(nf, p["gcn_hidden_dim"], p["gcn_dropout"]).to(device)
            hyb.gcn.load_state_dict(ck["state_dict"])
            hyb.gcn.eval()
            with open(clf_p, "rb") as f:
                hyb.clf = pickle.load(f)
            return "hybrid", hyb
        m = build_torch_model(name, nf, p).to(device)
        m.load_state_dict(ck["state_dict"])
        m.eval()
        return "torch", m
    with open(clf_p, "rb") as f:
        return "raw", pickle.load(f)
