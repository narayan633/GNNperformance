Default layout (relative to the repository root):
    data/Elliptic/      elliptic_txs_features.csv, elliptic_txs_classes.csv, elliptic_txs_edgelist.csv
    data/Ethereum/      eth_train.pkl, eth_val.pkl, eth_test.pkl  (TBNS graphs, K = 3)
    data/raw/           MulDiGraph.pkl  (full XBlock Ethereum graph, only for tbns_build_k.py)
    outputs/            everything the code writes


import os

ROOT = os.environ.get("CRYPTO_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATASET_DIR = os.path.join(ROOT, "data")
ELLIPTIC_RAW_DIR = os.path.join(DATASET_DIR, "Elliptic")
ETHEREUM_RAW_DIR = os.path.join(DATASET_DIR, "Ethereum")

ELLIPTIC_FEATURES_CSV = os.path.join(ELLIPTIC_RAW_DIR, "elliptic_txs_features.csv")
ELLIPTIC_CLASSES_CSV = os.path.join(ELLIPTIC_RAW_DIR, "elliptic_txs_classes.csv")
ELLIPTIC_EDGELIST_CSV = os.path.join(ELLIPTIC_RAW_DIR, "elliptic_txs_edgelist.csv")
ETHEREUM_PKL = {s: os.path.join(ETHEREUM_RAW_DIR, f"eth_{s}.pkl") for s in ("train", "val", "test")}

# Full XBlock Ethereum transaction graph, used only by tbns_build_k.py to build
# the K sensitivity variants.
ETHEREUM_MULTIDIGRAPH = os.environ.get("CRYPTO_ETH_GRAPH", os.path.join(DATASET_DIR, "raw", "MulDiGraph.pkl"))

# TBNS K sensitivity. K = 3 is the main setting 
TBNS_K_MAIN = 3
TBNS_K_VARIANTS = [1, 5]
K_DATASETS = [f"ethereum_k{k}" for k in TBNS_K_VARIANTS]


def ethereum_pkl_for_k(k):
    if k == TBNS_K_MAIN:
        return ETHEREUM_PKL
    return {s: os.path.join(DATASET_DIR, f"Ethereum_K{k}", f"eth_{s}.pkl") for s in ("train", "val", "test")}


def hpo_dataset_for(dataset):
    return "ethereum" if dataset.startswith("ethereum") else dataset

# Tuned settings (best HPO trial per model, dataset and split) shipped with the repository.
BEST_PARAMS_JSON = os.path.join(ROOT, "configs", "best_params.json")

OUT = os.path.join(ROOT, "outputs")
DIRS = {
    "data": os.path.join(OUT, "data"),
    "data_elliptic": os.path.join(OUT, "data", "elliptic"),
    "data_ethereum": os.path.join(OUT, "data", "ethereum"),
    "hpo": os.path.join(OUT, "hpo"),
    "runs": os.path.join(OUT, "runs"),
    "analysis": os.path.join(OUT, "analysis"),
    "timing": os.path.join(OUT, "timing"),
    "robustness": os.path.join(OUT, "robustness"),
    "explain": os.path.join(OUT, "explanation_stability"),
    "tbns": os.path.join(OUT, "tbns_evaluation"),
    "logs": os.path.join(OUT, "logs"),
    "registry": os.path.join(OUT, "registry"),
}
for _k in TBNS_K_VARIANTS:
    DIRS[f"data_ethereum_k{_k}"] = os.path.join(OUT, "data", f"ethereum_k{_k}")


def ensure_dirs():
    for p in DIRS.values():
        os.makedirs(p, exist_ok=True)


# ---------------------------------------------------------------------
# Datasets and splits
# ---------------------------------------------------------------------

DATASETS = ["elliptic", "ethereum"]

# Elliptic temporal split: train 1-34, val 35-42, test 43-49
ELLIPTIC_TRAIN_END = 34
ELLIPTIC_VAL_END = 42
ELLIPTIC_TOTAL_STEPS = 49
ELLIPTIC_N_FEATURES = 165

# Ethereum TBNS: train months 1-33, val 34-35, test 36-42
ETHEREUM_DROP_FEATURES = ["neighbor_fraud_ratio"]   # uses neighbour labels in train only
ETHEREUM_N_MONTHS = 42
MONTH_BOUNDARIES = [
    1438387200, 1441065600, 1443657600, 1446336000, 1448928000,
    1451606400, 1454284800, 1456790400, 1459468800, 1462060800,
    1464739200, 1467331200, 1470009600, 1472688000, 1475280000,
    1477958400, 1480550400, 1483228800, 1485907200, 1488326400,
    1491004800, 1493596800, 1496275200, 1498867200, 1501545600,
    1504224000, 1506816000, 1509494400, 1512086400, 1514764800,
    1517443200, 1519862400, 1522540800, 1525132800, 1527811200,
    1530403200, 1533081600, 1535760000, 1538352000, 1541030400,
    1543622400, 1546300800, 1548979200,
]  # 43 values = 42 months (month 1 = Aug 2015)


RANDOM_RATIOS = {
    "random70": (0.70, 0.15, 0.15),
    "random60": (0.60, 0.20, 0.20),
    "random80": (0.80, 0.10, 0.10),
}
CONDITIONS = ["temporal"] + list(RANDOM_RATIOS)

# Per-seed z-score normalisation under the random split.
# Elliptic: features are already standardised by the providers, so the random
# condition uses them as released.
# Ethereum: raw amounts and time spans, so normalisation is required.
NORMALISE_RANDOM = {"elliptic": False, "ethereum": True, **{d: True for d in K_DATASETS}}

# ---------------------------------------------------------------------
# Seeds and training protocol
# ---------------------------------------------------------------------

SEEDS_20 = [42, 52, 62, 72, 82, 92, 102, 112, 122, 132,
            142, 152, 162, 172, 182, 192, 202, 212, 222, 232]  
SEEDS_10 = SEEDS_20[:10]
SEEDS_EXPLAIN = SEEDS_20[:3]

FINAL_EPOCHS = 100
EVAL_EVERY = 5
PATIENCE = 7           
HPO_VERSION = "v2"
HPO_N_TRIALS = 50
HPO_N_SEEDS = 5
HPO_N_EPOCHS = 100      # same epochs and early stopping as final training
HPO_OVERRIDES = {       # EvolveGCN-H on Elliptic steps through 49 snapshots per epoch, so a smaller budget
    ("EvolveGCN-H", "elliptic", "temporal"): {"n_trials": 20, "n_seeds": 3},
    ("EvolveGCN-H", "elliptic", "random70"): {"n_trials": 20, "n_seeds": 3},
}


def hpo_budget(model, dataset, condition):
    o = HPO_OVERRIDES.get((model, dataset, condition), {})
    return {"n_trials": o.get("n_trials", HPO_N_TRIALS),
            "n_seeds": o.get("n_seeds", HPO_N_SEEDS),
            "n_epochs": o.get("n_epochs", HPO_N_EPOCHS)}


# ---------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------

MODELS = ["GCN", "GAT", "GCN+XGBoost", "GCN+KNN", "GCN+LSTM", "GNN+Autoencoder"]
TEMPORAL_MODELS = ["EvolveGCN-H"]
MAIN_MODELS = MODELS + TEMPORAL_MODELS

NOEDGE_MODELS = ["GCN+XGBoost-noEdge"]
ABLATION_MODELS = ["XGBoost-raw", "KNN-raw", "GNN+Autoencoder-noRecon"] + NOEDGE_MODELS

SEQUENCE_MODELS = {"EvolveGCN-H"}
HYBRID_MODELS = {"GCN+XGBoost", "GCN+KNN", "GCN+XGBoost-noEdge"}


def hpo_base_model(model):
   
    return {"GNN+Autoencoder-noRecon": "GNN+Autoencoder", "GCN+XGBoost-noEdge": "GCN+XGBoost"}.get(model, model)


def source_experiment(model):

    return "E5" if model in ABLATION_MODELS else "E2"
RAW_MODELS = {"XGBoost-raw", "KNN-raw"}

# Fixed settings of the earlier conference study (used in E3).
FIXED_PARAMS = {
    "GCN": {"hidden_dim": 64, "dropout": 0.3, "learning_rate": 0.01, "weight_decay": 5e-4},
    "GAT": {"hidden_dim": 64, "heads": 4, "dropout": 0.3, "learning_rate": 0.005, "weight_decay": 5e-4},
    "GCN+XGBoost": {"gcn_hidden_dim": 64, "gcn_dropout": 0.3, "learning_rate": 0.01, "weight_decay": 5e-4,
                    "xgb_n_estimators": 100, "xgb_max_depth": 5, "xgb_learning_rate": 0.1},
    "GCN+KNN": {"gcn_hidden_dim": 64, "gcn_dropout": 0.3, "learning_rate": 0.01, "weight_decay": 5e-4,
                "k_neighbors": 5},
    "GCN+LSTM": {"gcn_hidden": 64, "lstm_hidden": 64, "dropout": 0.3, "learning_rate": 0.01, "weight_decay": 5e-4},
    "GNN+Autoencoder": {"hidden_dim": 64, "latent_dim": 32, "dropout": 0.3, "alpha": 0.7,
                        "learning_rate": 0.01, "weight_decay": 5e-4},
}

# ---------------------------------------------------------------------
# Tuning conditions
# ---------------------------------------------------------------------
HPO_CONDITIONS = ["temporal", "random70"]   

# ---------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------
# Each experiment = (models, conditions, seeds, param_source)
# param_source: "tuned" (own condition), "fixed" (earlier study), or "transfer" (other condition).
EXPERIMENTS = {
    "E2": {"models": MAIN_MODELS, "conditions": ["temporal", "random70"], "seeds": SEEDS_20, "params": "tuned"},
    "E2b": {"models": MAIN_MODELS, "conditions": ["random60", "random80"], "seeds": SEEDS_20, "params": "tuned"},
    "E3": {"models": MODELS, "conditions": ["temporal", "random70"], "seeds": SEEDS_20, "params": "fixed"},
    "E4": {"models": MAIN_MODELS, "conditions": ["temporal", "random70"], "seeds": SEEDS_10, "params": "transfer"},
    "E5": {"models": ABLATION_MODELS, "conditions": ["temporal", "random70"], "seeds": SEEDS_20, "params": "tuned"},

    "E6K": {"models": MAIN_MODELS, "conditions": ["temporal", "random70"], "seeds": SEEDS_10, "params": "tuned",
            "datasets": K_DATASETS},
}


def experiment_datasets(exp):
    return EXPERIMENTS[exp].get("datasets", DATASETS)


def hpo_condition_for(condition):

    return "temporal" if condition == "temporal" else "random70"


def transfer_condition_for(condition):

    return "random70" if condition == "temporal" else "temporal"


# ---------------------------------------------------------------------
# Analysis settings
# ---------------------------------------------------------------------

ALERT_BUDGETS = [0.005, 0.01, 0.02, 0.05]
COST_RATIOS = [1, 5, 10, 25, 50, 100]          # false negative cost : false positive cost
BOOTSTRAP_N = 2000
ECE_BINS = 15
EDGE_DROP_RATES = [0.05, 0.10, 0.20]
FEATURE_NOISE_LEVELS = [0.01, 0.05]
ROBUSTNESS_REPEATS = 5
TIMING_FRACTIONS = [0.10, 0.25, 0.50, 0.75, 1.00]
TIMING_WARMUP = 1
TIMING_RUNS = 10

EXPLAIN = {
    "n_test_nodes": 50,
    "n_train_nodes": 50,
    "fraud_share": 0.5,
    "shap_nsamples": 1000,
    "shap_background": 20,    # weighted k-means summary; cost = nsamples x background model calls
    "lime_nsamples": 2000,
    "gnnexplainer_epochs": 100,
    "noise_level": 0.05,
    "noise_nodes": 20,
    "convergence_nodes": 10,
    "top_k": 5,
    "num_hops": 3,          # 3 hops keep 2-layer GCN normalisation exact
}
EXPLAIN_MODELS = MODELS   
