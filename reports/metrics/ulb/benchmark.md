| Model | Threshold | PR-AUC | ROC-AUC | Precision | Recall | F1 | Missed fraud | Alerts per catch |
|---|---|---|---|---|---|---|---|---|
| xgboost | 0.7166 | **0.8169** | 0.9681 | 0.9730 | 0.7579 | 0.8521 | 23 | 1.0 |
| autoencoder | 0.9444 | **0.4533** | 0.9374 | 0.5541 | 0.4316 | 0.4852 | 54 | 1.8 |
| isolation_forest | 0.7457 | **0.1143** | 0.9308 | 0.2283 | 0.2211 | 0.2246 | 74 | 4.4 |