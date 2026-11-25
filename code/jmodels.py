"""
jmodels.py

"""

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.utils import add_self_loops, degree


# =====================================================================
# End-to-end models
# =====================================================================

class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, dropout=0.3):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, 2)
        self.dropout = dropout

    def get_embeddings(self, x, edge_index):
        return F.relu(self.conv1(x, edge_index))

    def forward(self, x, edge_index):
        h = self.get_embeddings(x, edge_index)
        h = F.dropout(h, p=self.dropout, training=self.training)
        return F.log_softmax(self.conv2(h, edge_index), dim=1)


class GCNNoEdge(GCN):
    @staticmethod
    def _no_edges(x):
        return torch.empty((2, 0), dtype=torch.long, device=x.device)

    def get_embeddings(self, x, edge_index):
        return super().get_embeddings(x, self._no_edges(x))

    def forward(self, x, edge_index):
        return super().forward(x, self._no_edges(x))


class GAT(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, heads=4, dropout=0.3):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden_dim, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_dim * heads, 2, heads=1, concat=False, dropout=dropout)
        self.dropout = dropout

    def get_embeddings(self, x, edge_index):
        return F.elu(self.conv1(x, edge_index))

    def forward(self, x, edge_index):
        h = self.get_embeddings(x, edge_index)
        h = F.dropout(h, p=self.dropout, training=self.training)
        return F.log_softmax(self.conv2(h, edge_index), dim=1)


class GCN_LSTM(nn.Module):
    def __init__(self, in_dim, gcn_hidden=64, lstm_hidden=64, dropout=0.3, steps=3):
        super().__init__()
        self.conv1 = GCNConv(in_dim, gcn_hidden)
        self.lstm_cell = nn.LSTMCell(gcn_hidden, lstm_hidden)
        self.fc = nn.Linear(lstm_hidden, 2)
        self.dropout = dropout
        self.lstm_hidden = lstm_hidden
        self.steps = steps

    def get_embeddings(self, x, edge_index):
        return F.relu(self.conv1(x, edge_index))

    def forward(self, x, edge_index):
        h = F.dropout(self.get_embeddings(x, edge_index), p=self.dropout, training=self.training)
        hs = torch.zeros(h.size(0), self.lstm_hidden, device=h.device)
        cs = torch.zeros_like(hs)
        for _ in range(self.steps):
            hs, cs = self.lstm_cell(h, (hs, cs))
        return F.log_softmax(self.fc(hs), dim=1)


class GNN_Autoencoder(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, latent_dim=32, dropout=0.3):
        super().__init__()
        self.enc1 = GCNConv(in_dim, hidden_dim)
        self.enc2 = GCNConv(hidden_dim, latent_dim)
        self.dec1 = GCNConv(latent_dim, hidden_dim)
        self.dec2 = GCNConv(hidden_dim, in_dim)
        self.classifier = nn.Linear(latent_dim, 2)
        self.dropout = dropout

    def encode(self, x, edge_index):
        z = F.relu(self.enc1(x, edge_index))
        z = F.dropout(z, p=self.dropout, training=self.training)
        return self.enc2(z, edge_index)

    def get_embeddings(self, x, edge_index):
        return self.encode(x, edge_index)

    def decode(self, z, edge_index):
        return self.dec2(F.relu(self.dec1(z, edge_index)), edge_index)

    def forward(self, x, edge_index):
        return F.log_softmax(self.classifier(self.encode(x, edge_index)), dim=1)


def ae_combined_loss(model, x, edge_index, y, mask, class_weights, alpha):
    z = model.encode(x, edge_index)
    out = F.log_softmax(model.classifier(z), dim=1)
    cls = F.nll_loss(out[mask], y[mask], weight=class_weights)
    if alpha >= 1.0:
        return cls
    rec = F.mse_loss(model.decode(z, edge_index)[mask], x[mask])
    return alpha * cls + (1.0 - alpha) * rec


# =====================================================================
# EvolveGCN-H 
# =====================================================================

class TopKNodeSelector(nn.Module):
    def __init__(self, in_dim, k):
        super().__init__()
        self.k = k
        self.score = nn.Linear(in_dim, 1, bias=False)

    def forward(self, H):
        k = min(self.k, H.shape[0])
        s = torch.sigmoid(self.score(H).squeeze(-1))
        vals, idx = torch.topk(s, k)
        sel = H[idx] * vals.unsqueeze(-1)
        if k < self.k:
            sel = torch.cat([sel, torch.zeros(self.k - k, H.shape[1], device=H.device, dtype=H.dtype)], 0)
        return sel


class EvolveGCNHLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.selector = TopKNodeSelector(in_dim, k=in_dim)
        self.gru = nn.GRUCell(in_dim, out_dim)
        self.W_init = nn.Parameter(torch.empty(in_dim, out_dim))
        nn.init.xavier_uniform_(self.W_init)

    def evolve(self, H, W_prev):
        return self.gru(self.selector(H), W_prev)

    @staticmethod
    def propagate(x, edge_index, W):
        ei, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        row, col = ei
        deg = degree(col, x.size(0), dtype=x.dtype)
        dis = deg.pow(-0.5)
        dis[torch.isinf(dis)] = 0
        norm = dis[row] * dis[col]
        sup = x @ W
        out = torch.zeros_like(sup)
        out.index_add_(0, col, sup[row] * norm.unsqueeze(-1))
        return out


class EvolveGCNH(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, dropout=0.3):
        super().__init__()
        self.l1 = EvolveGCNHLayer(in_dim, hidden_dim)
        self.l2 = EvolveGCNHLayer(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 2)
        self.dropout = dropout

    def run_sequence(self, snapshots):
        """snapshots: chronological list of Data. Returns one log-prob tensor per snapshot."""
        W1 = self.l1.W_init
        W2 = self.l2.W_init
        outs = []
        for d in snapshots:
            W1 = self.l1.evolve(d.x, W1)
            h1 = F.relu(self.l1.propagate(d.x, d.edge_index, W1))
            h1 = F.dropout(h1, p=self.dropout, training=self.training)
            W2 = self.l2.evolve(h1, W2)
            h2 = F.relu(self.l2.propagate(h1, d.edge_index, W2))
            h2 = F.dropout(h2, p=self.dropout, training=self.training)
            outs.append(F.log_softmax(self.classifier(h2), dim=1))
        return outs


# =====================================================================
# Hybrids and graph-free baselines
# =====================================================================

def _expand_proba(raw, classes):
    full = np.zeros((raw.shape[0], 2))
    for i, c in enumerate(classes):
        full[:, int(c)] = raw[:, i]
    return full


class HybridGCN:
    """GCN encoder trained end to end with early stopping, then a downstream classifier on embeddings."""

    def __init__(self, in_dim, params, kind, seed, no_edge=False):
        self.kind = kind
        self.params = params
        self.seed = seed
        self.no_edge = no_edge
        enc = GCNNoEdge if no_edge else GCN
        self.gcn = enc(in_dim, params["gcn_hidden_dim"], params["gcn_dropout"])
        self.clf = None

    def make_classifier(self):
        if self.kind == "xgb":
            from xgboost import XGBClassifier
            return XGBClassifier(n_estimators=int(self.params["xgb_n_estimators"]),
                                 max_depth=int(self.params["xgb_max_depth"]),
                                 learning_rate=float(self.params["xgb_learning_rate"]),
                                 tree_method="hist", eval_metric="logloss", random_state=self.seed, n_jobs=4)
        from sklearn.neighbors import KNeighborsClassifier
        return KNeighborsClassifier(n_neighbors=int(self.params["k_neighbors"]), n_jobs=4)

    def embeddings(self, data):
        self.gcn.eval()
        with torch.no_grad():
            return self.gcn.get_embeddings(data.x, data.edge_index).cpu().numpy()

    def fit_classifier(self, emb, y):
        self.clf = self.make_classifier()
        if self.kind == "knn" and self.clf.n_neighbors > len(y):
            self.clf.n_neighbors = max(1, len(y))
        self.clf.fit(emb, y)

    def predict_proba_emb(self, emb):
        raw = self.clf.predict_proba(emb)
        return raw[:, 1] if raw.shape[1] == 2 else _expand_proba(raw, self.clf.classes_)[:, 1]


class RawBaseline:
    def __init__(self, params, kind, seed):
        self.params = params
        self.kind = kind
        self.seed = seed
        self.clf = None

    def fit(self, X, y):
        if self.kind == "xgb":
            from xgboost import XGBClassifier
            spw = float((y == 0).sum() / max((y == 1).sum(), 1)) if self.params.get("scale_pos_weight_on") else 1.0
            self.clf = XGBClassifier(n_estimators=int(self.params["xgb_n_estimators"]),
                                     max_depth=int(self.params["xgb_max_depth"]),
                                     learning_rate=float(self.params["xgb_learning_rate"]),
                                     scale_pos_weight=spw, tree_method="hist", eval_metric="logloss",
                                     random_state=self.seed, n_jobs=4)
        else:
            from sklearn.neighbors import KNeighborsClassifier
            self.clf = KNeighborsClassifier(n_neighbors=min(int(self.params["k_neighbors"]), len(y)),
                                            weights=self.params.get("knn_weights", "uniform"), n_jobs=4)
        self.clf.fit(X, y)

    def predict_proba(self, X):
        raw = self.clf.predict_proba(X)
        return raw[:, 1] if raw.shape[1] == 2 else _expand_proba(raw, self.clf.classes_)[:, 1]


# =====================================================================
# Factory
# =====================================================================

def build_torch_model(name, in_dim, p):
    if name == "GCN":
        return GCN(in_dim, p["hidden_dim"], p["dropout"])
    if name == "GAT":
        return GAT(in_dim, p["hidden_dim"], p["heads"], p["dropout"])
    if name == "GCN+LSTM":
        return GCN_LSTM(in_dim, p["gcn_hidden"], p["lstm_hidden"], p["dropout"])
    if name in ("GNN+Autoencoder", "GNN+Autoencoder-noRecon"):
        return GNN_Autoencoder(in_dim, p["hidden_dim"], p["latent_dim"], p["dropout"])
    if name == "EvolveGCN-H":
        return EvolveGCNH(in_dim, p["hidden_dim"], p["dropout"])
    raise ValueError(f"not a torch model: {name}")


def clone_state(model):
    return copy.deepcopy(model.state_dict())
