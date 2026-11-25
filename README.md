# When does the graph help? GNNs for blockchain fraud detection

Code, tuned settings and result tables for the paper *When does the graph help? Predictive performance, robustness and reliability of graph neural networks for blockchain fraud detection*.

The study compares seven GNN and hybrid models (GCN, GAT, GCN+XGBoost, GCN+KNN, GCN+LSTM, GNN+Autoencoder and EvolveGCN-H) with tuned graph-free baselines on the Elliptic Bitcoin dataset and an Ethereum phishing graph built with temporally bounded neighbourhood sampling (TBNS). It covers predictive performance, graph construction, inference cost, alert-level reliability, robustness to feature noise and explanation stability.

> Citation: to be added after publication.

## Repository layout


configs/
    best_params.json             tuned settings, one entry per model|dataset|split (best Optuna trial)
    search_spaces.md             tuning protocol and search spaces
src/
    jconfig.py                   paths, seeds, splits, models and experiment definitions
    jutils.py                    atomic writes, seeding, logging, run folders
    jdata.py                     graph loading and random splits
    jmodels.py                   GCN, GAT, GCN+LSTM, GNN+Autoencoder, EvolveGCN-H, hybrids, raw baselines
    jtrain.py                    training and evaluation of one run
    jmetrics.py                  AUC-PRC, F1, MCC, ECE, Brier, alert budgets, expected cost
    params.py                    settings of each run, read from configs/best_params.json
    data_build.py                Elliptic preprocessing and conversion of the TBNS graphs
    tbns_build_k.py              TBNS graphs with K = 1 and K = 5
    tbns_evaluation.py           TBNS audit and structure tests 
    inference_timing.py          latency, throughput and memory against graph size 
    perturbation_robustness.py   edge drop and feature noise at inference 
    encoder control without the tree-setting.py        raw XGBoost with the tree settings of the no-edge pipeline 
    explanation_stability.py     KernelSHAP, LIME, feature mask and GNNExplainer 
    explain_ten_seeds.py         extends the explanation experiment to ten training seeds 
    analysis_reliability.py      reliability tables from the saved predictions 
    analysis_statistics.py       statistical tests and the design synthesis
    results/                         the numbers behind every table and figure


## Setup

Python 3.12 with a CUDA build of PyTorch. The reported results were produced with the versions in `requirements.txt` on one NVIDIA RTX 5080 (16 GB) under Windows 11.


## Data


```
data/Elliptic/   elliptic_txs_features.csv, elliptic_txs_classes.csv, elliptic_txs_edgelist.csv
data/Ethereum/   eth_train.pkl, eth_val.pkl, eth_test.pkl
data/raw/        MulDiGraph.pkl [for ethereum]
```

- **Elliptic.** The Elliptic Data Set (Weber et al., 2019), available on Kaggle.
- **Ethereum.** The XBlock Ethereum phishing transaction network (`MulDiGraph.pkl`). The TBNS split graphs `eth_*.pkl` (K = 3; months 1 to 33, 34 to 35 and 36 to 42) are built from this network with the procedure in Algorithm 1 of the paper. `tbns_build_k.py` rebuilds the neighbourhood step for K = 1 and K = 5 from `MulDiGraph.pkl`, and first checks that it reproduces the K = 3 graphs exactly.

## Reproducing the results

All commands run from `src/`. Every script skips work that is already finished, so it can be stopped and started again.

```
python data_build.py                     # graphs used by every experiment
python tbns_build_k.py                   # TBNS K = 1 and K = 5, then run data_build.py again

python train_one.py --exp E2             # tuned runs, 7 models x 2 datasets x 2 splits x 20 seeds
python train_one.py --exp E2b            # random 60/20/20 and 80/10/10
python train_one.py --exp E3             # fixed settings of the earlier conference study
python train_one.py --exp E4             # settings transferred between splits
python train_one.py --exp E5             # raw XGBoost, raw KNN, GNN+Autoencoder without reconstruction,
                                         # and GCN+XGBoost without edges (the no-edge pipeline)
python train_one.py --exp E6K            # TBNS K = 1 and K = 5

python tbns_evaluation.py                # RQ2
python inference_timing.py               # RQ3 (run alone on the machine)
python perturbation_robustness.py        # RQ4
python c1_matched_control.py             # matched control (after E5 and the perturbation tests)
python explanation_stability.py          # RQ5, seeds 42, 52, 62 with the repeat floor
python explanation_stability.py --gnnx-only
python explain_ten_seeds.py              # RQ5, seeds 72 to 132, then the ten-seed tables

python analysis_reliability.py           # reliability tables
python analysis_statistics.py            # tests and the design synthesis
```

## Result tables in `results/`

Per-seed and per-node outputs are large, so the tables here are aggregated over seeds (mean, standard deviation and count). Running the scripts above writes the full per-seed files to `outputs/`.

| File | Content | Paper |
| --- | --- | --- |
| `runs.csv` | All 2,480 training runs with settings, threshold, validation and test metrics | Tables 5, A2, Figure 3 |
| `hpo/hpo_sensitivity.json` | Best value, top trials and parameter importance of every search (the selected settings are in `configs/best_params.json`, Table A7) | Section 4.4 |
| `analysis/tuning_effect.csv`, `stats_tuning.csv` | Tuned against fixed and transferred settings | Section 6.1, Figure 3 |
| `analysis/stats_pairwise_models.csv`, `stats_ranks.csv` | Pairwise Wilcoxon tests with Holm correction, bootstrap, Friedman test | Section 6.2 |
| `analysis/stats_ablation.csv`, `stats_encoder_control.csv` | Component ablations and the encoder control under noise | Sections 6.3, 9.5 |
| `analysis/complementarity_by_pair.csv`, `degree_by_model.csv`, `per_step_by_model.csv` | Error overlap, results by degree, Elliptic per time step | Section 6.4, Figure A1, Tables A3, A4 |
| `analysis/seed_stability.csv`, `ratio_stability.csv`, `churn.csv` | Seed and split-ratio stability, alert agreement between seeds | Section 9.1, Figure 6 |
| `analysis/threshold_transfer_by_model.csv`, `alert_budget_by_model.csv` | Threshold transfer and precision at alert budgets | Section 9.2, Table 8 |
| `analysis/calibration_by_model.csv`, `calibration_bins.csv` | ECE, Brier score and reliability bins | Section 9.3, Table 8 |
| `analysis/cost_by_model.csv` | Normalised misclassification cost for ratios 1 to 100 | Section 9.4, Table 8 |
| `analysis/tbns_k_sensitivity.csv`, `stats_tbns_k.csv` | TBNS with K = 1, 3 and 5 | Section 7.2, Figure 4 |
| `analysis/synthesis_*.csv` | Design synthesis, with SHAP overlap over ten seeds | Section 11.1, Figure 9, Table A6 |
| `tbns/` | TBNS audit, graph sizes for every K, K = 3 reproduction check and structure tests | Sections 3.2, 7.1, Tables 6, A1 |
| `datasets/*_metadata.json` | Graph sizes, labelled and fraud counts, preprocessing | Table 2 |
| `timing/summary.csv` | Latency, throughput and memory on GPU and CPU, 10% to 100% of the graph | Table 7, Figure 5 |
| `robustness/summary_by_model.csv` | Edge drop and feature noise, mean over seeds of the five draws per seed | Figure 7, Table A5 |
| ` encoder control without the tree-setting/` | Matched raw XGBoost control, summary and paired tests | Sections 6.3, 9.5 |
| `explanation_10seeds/` | Explanation stability over ten seeds, the repeat floor, noise, explainer agreement, edges, runtime | Section 10, Figure 8 |
| `explanation/nodes/*.json` | The node IDs explained in each dataset and split | Section 5.5 |

Training and wall times in `runs.csv` were measured while several runs shared one GPU. Use `timing/summary.csv` for cost comparisons.



