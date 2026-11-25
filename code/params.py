"""
params.py
=========
Hyperparameters for every run, read from configs/best_params.json (the best
Optuna trial for each model, dataset and split; 50 trials, 5 seeds, 100 epochs).

"""

import jconfig as C
from jutils import read_json

_CACHE = None


def _table():
    global _CACHE
    if _CACHE is None:
        _CACHE = read_json(C.BEST_PARAMS_JSON)
        if _CACHE is None:
            raise FileNotFoundError(f"{C.BEST_PARAMS_JSON} not found")
    return _CACHE


def get_best_params(model, dataset, condition):
    base = C.hpo_base_model(model)
    key = f"{base}|{C.hpo_dataset_for(dataset)}|{condition}"
    if key not in _table():
        raise KeyError(f"no tuned settings for {key}")
    p = dict(_table()[key]["best_params"])
    if model == "GNN+Autoencoder-noRecon":
        p["alpha"] = 1.0
    return p


def params_for(exp, model, dataset, condition):

    src = C.EXPERIMENTS[exp]["params"]
    if src == "fixed":
        return dict(C.ICBC_FIXED_PARAMS[model]), {"source": "icbc_fixed"}
    hc = C.hpo_condition_for(condition) if src == "tuned" else C.transfer_condition_for(condition)
    base = C.hpo_base_model(model)
    entry = _table()[f"{base}|{C.hpo_dataset_for(dataset)}|{hc}"]
    return get_best_params(model, dataset, hc), {
        "source": src, "hpo_condition": hc, "hpo_model": base, "hpo_dataset": C.hpo_dataset_for(dataset),
        "hpo_best_val": entry["best_val_auc_prc"]}
