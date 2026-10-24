| Arm | SMOTE | Threshold | PR-AUC | Precision | Recall | F1 | Missed fraud | Alerts per catch |
|---|---|---|---|---|---|---|---|---|
| xgboost+smote | yes | 0.8507 | **0.5594** | 0.8919 | 0.3708 | 0.5238 | 56 | 1.1 |
| xgboost+weighted | no | 0.9034 | **0.5355** | 0.8182 | 0.4045 | 0.5414 | 53 | 1.2 |
| autoencoder+weighted | no | 0.9832 | **0.2835** | 1.0000 | 0.2022 | 0.3364 | 71 | 1.0 |
| isolation_forest+weighted | no | 0.7225 | **0.0244** | 0.4000 | 0.0225 | 0.0426 | 87 | 2.5 |