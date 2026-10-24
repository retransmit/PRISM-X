| Arm | SMOTE | Threshold | PR-AUC | Precision | Recall | F1 | Missed fraud | Alerts per catch |
|---|---|---|---|---|---|---|---|---|
| xgboost+weighted | no | 0.7166 | **0.8169** | 0.9730 | 0.7579 | 0.8521 | 23 | 1.0 |
| xgboost+smote | yes | 0.8287 | **0.8155** | 0.9722 | 0.7368 | 0.8383 | 25 | 1.0 |
| autoencoder+weighted | no | 0.9444 | **0.4533** | 0.5541 | 0.4316 | 0.4852 | 54 | 1.8 |
| isolation_forest+weighted | no | 0.7457 | **0.1143** | 0.2283 | 0.2211 | 0.2246 | 74 | 4.4 |