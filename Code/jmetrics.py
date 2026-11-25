import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss,
                             f1_score, matthews_corrcoef, precision_recall_curve, precision_score, recall_score,
                             roc_auc_score)


def best_f1_threshold(y, p):

    y = np.asarray(y)
    p = np.asarray(p, dtype=np.float64)
    if y.min() == y.max():
        return 0.5
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0.0)
    i = int(np.nanargmax(f1[:-1])) if len(thr) else 0
    return float(thr[i]) if len(thr) else 0.5


def ece(y, p, n_bins=15):
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, bins[1:-1]), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(total)


def threshold_metrics(y, p, thr):
    yp = (np.asarray(p) >= thr).astype(int)
    return {
        "f1_fraud": float(f1_score(y, yp, pos_label=1, zero_division=0)),
        "precision": float(precision_score(y, yp, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y, yp, pos_label=1, zero_division=0)),
        "f1_macro": float(f1_score(y, yp, average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(y, yp)) if len(np.unique(yp)) > 1 or len(np.unique(y)) > 1 else 0.0,
        "balanced_accuracy": float(balanced_accuracy_score(y, yp)) if len(np.unique(y)) > 1 else 0.0,
        "accuracy": float(accuracy_score(y, yp)),
        "n_alerts": int(yp.sum()),
    }


def compute_metrics(y, p, thr, n_bins=15):
    y = np.asarray(y).astype(int)
    p = np.clip(np.asarray(p, dtype=np.float64), 0.0, 1.0)
    both = len(np.unique(y)) > 1
    prev = float(y.mean()) if len(y) else 0.0
    m = {
        "n": int(len(y)), "n_fraud": int(y.sum()), "prevalence": prev,
        "auc_prc": float(average_precision_score(y, p)) if both else float("nan"),
        "auc_roc": float(roc_auc_score(y, p)) if both else float("nan"),
        "brier": float(brier_score_loss(y, p)) if len(y) else float("nan"),
        "ece": ece(y, p, n_bins) if len(y) else float("nan"),
        "threshold": float(thr),
    }
    m["auc_prc_lift"] = m["auc_prc"] / prev if both and prev > 0 else float("nan")
    m.update(threshold_metrics(y, p, thr))
    at05 = threshold_metrics(y, p, 0.5)
    m.update({f"{k}@0.5": v for k, v in at05.items()})
    return m


def precision_recall_at_budget(y, p, budget):
    """Precision and recall among the top `budget` share of scored nodes."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=np.float64)
    k = max(1, int(round(budget * len(y))))
    top = np.argsort(-p, kind="stable")[:k]
    tp = int(y[top].sum())
    return {"k": k, "precision": tp / k, "recall": tp / max(int(y.sum()), 1), "tp": tp}


def expected_cost(y, p, thr, fn_cost, fp_cost=1.0):

    y = np.asarray(y).astype(int)
    yp = (np.asarray(p) >= thr).astype(int)
    fn = int(((y == 1) & (yp == 0)).sum())
    fp = int(((y == 0) & (yp == 1)).sum())
    cost = fn * fn_cost + fp * fp_cost
    base = max(int(y.sum()) * fn_cost, 1e-12)
    return {"fn": fn, "fp": fp, "cost": float(cost), "normalised_cost": float(cost / base)}


def cost_optimal_threshold(y, p, fn_cost, fp_cost=1.0):

    Exact search over every distinct score (plus 'flag nothing'), O(n log n)."""
    y = np.asarray(y).astype(np.int64)
    p = np.asarray(p, dtype=np.float64)
    if len(p) == 0:
        return 0.5
    order = np.argsort(-p, kind="stable")
    ps, ys = p[order], y[order]
    last = np.r_[np.nonzero(np.diff(ps))[0], len(ps) - 1]      # last position of each distinct score
    n_pos = int(ys.sum())
    tp = np.cumsum(ys)[last]
    fp = np.cumsum(1 - ys)[last]
    costs = (n_pos - tp) * fn_cost + fp * fp_cost
    none_cost = n_pos * fn_cost
    i = int(np.argmin(costs))
    if none_cost < costs[i]:
        return float(ps[0] + 1e-9)
    return float(ps[last[i]])
