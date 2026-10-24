| Model | Held-out scenario | Seeds | Held-out fraud rows | Held-out recall | PR-AUC |
|---|---|---|---|---|---|
| autoencoder | INTENT_OMEGA_BISECT_DRAIN | 5 | 166 | 0.137 [0.064, 0.196] | 0.1705 [0.1420, 0.1908] |
| autoencoder | INTENT_OMEGA_HARVEST | 5 | 165 | 0.152 [0.035, 0.268] | 0.1638 [0.1351, 0.1874] |
| autoencoder | INTENT_OMEGA_PROBE | 5 | 252 | 0.004 [0.000, 0.013] | 0.1705 [0.1427, 0.1930] |
| isolation_forest | INTENT_OMEGA_BISECT_DRAIN | 5 | 166 | 0.000 [0.000, 0.000] | 0.0169 [0.0134, 0.0202] |
| isolation_forest | INTENT_OMEGA_HARVEST | 5 | 165 | 0.000 [0.000, 0.000] | 0.0171 [0.0127, 0.0216] |
| isolation_forest | INTENT_OMEGA_PROBE | 5 | 252 | 0.000 [0.000, 0.000] | 0.0174 [0.0131, 0.0224] |
| xgboost | INTENT_OMEGA_BISECT_DRAIN | 5 | 166 | 0.165 [0.076, 0.245] | 0.5341 [0.4958, 0.5631] |
| xgboost | INTENT_OMEGA_HARVEST | 5 | 165 | 0.196 [0.117, 0.263] | 0.5008 [0.4713, 0.5312] |
| xgboost | INTENT_OMEGA_PROBE | 5 | 252 | 0.008 [0.000, 0.016] | 0.3363 [0.2800, 0.3862] |
