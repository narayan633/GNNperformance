import glob
import os

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data

import jconfig as C


def _load_npz(path):
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


def to_data(g, device="cpu"):
    d = Data(x=torch.as_tensor(g["x"], dtype=torch.float32),
             edge_index=torch.as_tensor(g["edge_index"], dtype=torch.long),
             y=torch.as_tensor(g["y"], dtype=torch.long))
    d.labeled_mask = torch.as_tensor(g["labeled"], dtype=torch.bool)
    d.has_features = torch.as_tensor(g["has_features"], dtype=torch.bool)
    d.time = torch.as_tensor(g["time"], dtype=torch.long)
    d.node_id = np.asarray(g["node_id"]).astype(str)     # kept as numpy strings (not moved to GPU)
    return d.to(device)


def node_degree(edge_index, n):
    ei = np.asarray(edge_index)
    return (np.bincount(ei[0], minlength=n) + np.bincount(ei[1], minlength=n)).astype(np.int64)


class DatasetBundle:

    def __init__(self, dataset):
        self.dataset = dataset
        base = C.DIRS[f"data_{dataset}"]
        if not os.path.exists(os.path.join(base, "metadata.json")):
            raise FileNotFoundError(f"{base} not built. Run: python data_build.py --dataset {dataset}")
        self.base = base
        self.flat = {s: _load_npz(os.path.join(base, f"temporal_{s}.npz")) for s in ("train", "val", "test")}
        self.full = _load_npz(os.path.join(base, "full_graph.npz"))
        self._snaps = {}

    def snapshots(self, which):
        """which in {'train','val','test','full'}: chronological list of graphs."""
        if which not in self._snaps:
            d = os.path.join(self.base, f"temporal_{which}_snapshots" if which != "full" else "full_snapshots")
            files = sorted(glob.glob(os.path.join(d, "t*.npz")))
            if not files:
                raise FileNotFoundError(f"No snapshots in {d}")
            self._snaps[which] = [_load_npz(f) for f in files]
        return self._snaps[which]

    @property
    def n_features(self):
        return int(self.full["x"].shape[1])


# ---------------------------------------------------------------------
# Random split (shared by every model)
# ---------------------------------------------------------------------

def random_split(full_graph, seed, condition):
    """Returns dict of sets of node IDs (strings) for train/val/test."""
    tr, va, te = C.RANDOM_RATIOS[condition]
    lab = full_graph["labeled"]
    ids = np.asarray(full_graph["node_id"]).astype(str)[lab]
    y = full_graph["y"][lab]
    order = np.argsort(ids, kind="stable")          # stable order, independent of file layout
    ids, y = ids[order], y[order]
    id_tr, id_tmp, _, y_tmp = train_test_split(ids, y, train_size=tr, random_state=seed, stratify=y)
    val_share = va / (va + te)
    id_va, id_te = train_test_split(id_tmp, train_size=val_share, random_state=seed, stratify=y_tmp)
    return {"train": set(id_tr.tolist()), "val": set(id_va.tolist()), "test": set(id_te.tolist())}


def masks_from_ids(node_ids, id_sets, labeled):
    node_ids = np.asarray(node_ids).astype(str)
    out = {}
    for k, s in id_sets.items():
        m = np.fromiter((n in s for n in node_ids), dtype=bool, count=len(node_ids))
        out[k] = m & labeled
    return out


def normalise_random(x, train_mask, has_features):
    fit = train_mask & has_features
    mu = x[fit].mean(axis=0)
    sd = x[fit].std(axis=0)
    sd[sd < 1e-8] = 1.0
    xn = ((x - mu) / sd).astype(np.float32)
    xn[~has_features] = 0.0
    return xn, mu, sd
