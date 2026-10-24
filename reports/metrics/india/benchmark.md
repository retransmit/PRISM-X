| Model | Threshold | PR-AUC | ROC-AUC | Precision | Recall | F1 | Missed fraud | Alerts per catch | Precision at 1 in 80,847 | FPR per 10k | Value above floor caught |
|---|---|---|---|---|---|---|---|---|---|---|---|
| xgboost | 0.9034 | **0.5355** | 0.9762 | 0.8182 | 0.4045 | 0.5414 | 53 | 1.2 | 0.0111 | 4.5 | 0.533 |
| autoencoder | 0.9832 | **0.2835** | 0.8184 | 1.0000 | 0.2022 | 0.3364 | 71 | 1.0 | 1.0000 | 0.0 | 0.955 |
| isolation_forest | 0.7225 | **0.0244** | 0.7484 | 0.4000 | 0.0225 | 0.0426 | 87 | 2.5 | 0.0017 | 1.7 | 0.006 |