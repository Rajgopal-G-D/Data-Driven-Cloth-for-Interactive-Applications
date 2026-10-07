## Test-set results

| Model | Test MSE | Mean sq. per-vertex err | Relative err (%) | Best ex. | Worst ex. | Trainable params | Inference (ms/mesh) |
|---|---|---|---|---|---|---|---|
| 0_linear | 6.615e-03 | 1.984e-02 | 23.32 | 1.21e-02 | 3.17e-02 | 371,070 | 0.0369 |
| 1_offsets (best on val) | 1.375e-04 | 4.126e-04 | 0.48 | 1.27e-04 | 6.90e-03 | 1,713,390 | 0.0851 |
| 2_pca256 | 1.451e-04 | 4.354e-04 | 0.51 | 9.96e-05 | 7.57e-03 | 106,112 | 0.1031 |
| 3_pca128 | 1.347e-04 | 4.042e-04 | 0.48 | 1.13e-04 | 5.45e-03 | 73,216 | 0.0827 |
| 4_pca64 | 1.771e-03 | 5.313e-03 | 6.24 | 2.80e-03 | 1.37e-02 | 56,768 | 0.0965 |

Best model (by validation MSE): **1_offsets**; improvement over linear baseline in test MSE: **97.9%**.
Throughput of best model at batch size 1: ~11749 meshes/s on cpu.