# Data-Driven Clothing for Interactive Applications

**UE24CS352A Machine Learning - Mini Project**
**Team:** <Name 1 (SRN)>, <Name 2 (SRN)>
**Problem statement no.:** <number>
**Repository:** <GitHub URL>

Reimplementation of the pipeline from Xue & Wu, *Data-Driven Clothing for Interactive Applications*:
predict the shape of a loose coat (vertex offsets from a skinned mesh) from joint rotations, using
PCA and a fully-connected neural network, and compare against a linear-regression baseline.

> **Important:** the paper's dataset (Blender skinning + PhysBAM cloth simulation) is not publicly
> available. All experiments here use a **synthetic stand-in dataset** with the same input/output
> shapes. Absolute error values are therefore **not comparable with the paper's**; the comparison
> *between models* is what this project reports.

## Problem

Given 14 joint-rotation quaternions (14 x 4 = 56 values), predict the coat's per-vertex offsets
(2,170 vertices x 3 = 6,510 values) so that a coat shape can be produced in real time instead of by a
slow physics simulation.

## Dataset (synthetic)

| Item | Value |
|---|---|
| Examples generated | 10,000 |
| Input | 56-d (14 unit quaternions, w >= 0) |
| Output | 6,510-d (2,170 vertices x 3) |
| Target construction | linear (skinning-like) term + smooth nonlinear (`tanh`) term + small Gaussian noise |
| Cleaning | drop the 2% of examples with the largest mean squared offset distance (stands in for the paper's removal of faulty simulations) |
| Split | 6,000 train / 2,000 validation / 1,800 test (after cleaning) |

The generator is `make_dataset()` in `cloth_pipeline.py` (seed 0), so the data is exactly reproducible.
To use real data, replace `load_dataset()` with a loader returning `X` (N, 56) and `Y` (N, 6510).

## Method

1. **PCA** of the training offsets (via SVD, fitted on the training set only): `x = sum_l c_l u_l + mu`.
2. **Linear regression baseline**: a network with no hidden layers (56 -> 6,510).
3. **Fully-connected network**: ReLU 128 -> 256, then either
   - the 6,510 offsets directly, or
   - *k* PCA coefficients followed by a **frozen "PC layer"** (weights = principal components, bias = mean) that reconstructs the offsets. Tested with k = 256, 128, 64.
4. **Training**: MSE on the reconstructed offsets, Adam, batch size 32, 1,000 epochs, learning rate 1e-3, seed 0. The model is selected on **validation** MSE; the test set is used only for reporting.

## Repository structure

| File | Purpose |
|---|---|
| `cloth_mini_project.ipynb` | Everything in one notebook (runs on Google Colab) |
| `cloth_pipeline.py` | Shared code: data generator, PCA, models, training loop, timing |
| `prepare_data.py` | Step 2: generate data, clean, save `data/dataset.npz`, distance histogram |
| `pca_analysis.py` | Step 3: explained variance and reconstruction error |
| `train_models.py` | Step 4: train the five models, save checkpoints and `results.json` |
| `evaluate.py` | Step 5: test-set table, timing, error plots |
| `results.json`, `results_table.md` | Final results |
| `figures/` | Generated plots |
| `experiments/` | Backups of earlier runs (learning-rate and scaling experiments) |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

**Option A - Google Colab (simplest):** upload `cloth_mini_project.ipynb`, enable a GPU
(Runtime > Change runtime type), and choose Runtime > Run all. Set `QUICK = True` in the first code
cell for a one-minute smoke test.

**Option B - scripts:**

```bash
python prepare_data.py                              # -> data/dataset.npz, figures/distance_histogram.png
python pca_analysis.py                              # -> figures/pca_analysis.png, pca_stats.json
python train_models.py --epochs 1000 --lr 1e-3      # -> results.json, checkpoints/, figures/training_curves.png
python evaluate.py                                  # -> results_table.md, figures/evaluation.png
```

Use `--epochs 3` on `train_models.py` first as a quick check. The default learning rate of
`train_models.py` is 1e-4; the reported results use **1e-3** (see below).

## Results

All five models use identical settings (lr 1e-3, 1,000 epochs, batch 32, seed 0). Test set: 1,800 examples.

| Model | Test MSE | Mean sq. per-vertex err | Relative err (%) | Trainable params |
|---|---|---|---|---|
| 0 Linear regression | 6.615e-3 | 1.984e-2 | 23.32 | 371,070 |
| 1 MLP -> offsets (best on validation) | 1.375e-4 | 4.126e-4 | 0.48 | 1,713,390 |
| 2 MLP -> 256 PCA coeffs | 1.451e-4 | 4.354e-4 | 0.51 | 106,112 |
| 3 MLP -> 128 PCA coeffs | **1.347e-4** | **4.042e-4** | 0.48 | 73,216 |
| 4 MLP -> 64 PCA coeffs | 1.771e-3 | 5.313e-3 | 6.24 | 56,768 |

* Best model by validation MSE: **1 (direct offsets)**; it improves test MSE over the linear baseline by **97.9%** (relative error 23.3% -> 0.48%).
* Models 1, 2 and 3 are **comparable** in our setting: single run, one seed, differences of a few percent. We do not claim a ranking among them.
* Inference takes roughly 0.1 ms per mesh on CPU (about 10,000 meshes/s at batch size 1). Timings varied between runs (0.04 to 0.2 ms), so only the order of magnitude is meaningful.

### Learning-rate sensitivity (test MSE)

| Model | lr 1e-4 | lr 1e-3 |
|---|---|---|
| Linear regression | 6.49e-3 | 6.62e-3 |
| MLP -> offsets | 5.03e-4 | 1.38e-4 |
| MLP -> 256 PCA | 2.04e-3 | 1.45e-4 |
| MLP -> 128 PCA | 2.44e-3 | 1.35e-4 |
| MLP -> 64 PCA | 2.90e-3 | 1.77e-3 |

At lr 1e-4 the PCA models looked much worse than direct prediction even after 1,000 epochs, with
train, validation and test errors close together (underfitting, not overfitting). A tenfold
higher learning rate removed most of the gap. We also tried standardising the PCA coefficients
(`--scale`) and it made no difference, so we did not use it.

### PCA analysis

* Components needed for 90% / 99% of the variance: <k90> / <k99> (see `figures/pca_analysis.png`).
* Reconstruction floor (mean squared per-vertex error on training data): 4.8e-3 at k = 64 and 2.9e-6 at k = 128.
  This explains model 4: 64 components cannot represent the data, regardless of training.

## Discussion

* Neural networks predict the coat offsets far better than the linear baseline, because the data
  contains nonlinear structure that a linear model cannot capture. This agrees with the paper.
* PCA with 128 coefficients matches direct prediction with about 23x fewer trainable parameters
  (73k vs 1.7M). PCA with only 64 coefficients is limited by reconstruction error.
* **Difference from the paper:** the paper reports 128 PCA coefficients clearly beating direct
  prediction, with 256 overfitting. We do not observe a separation between 1, 2 and 3. Possible
  reasons (untested): our data is synthetic and noise-free in structure, and results come from a
  single seed.

## Limitations

* Synthetic data; no real cloth simulation, no mesh rendering, and no temporal coherence (as in the paper).
* One random seed per model; no confidence intervals.
* We did not run the paper's PhysBAM simulator, so we make no speed-up claim relative to simulation.
  The paper itself reports about 180 s per simulated coat; our figures cover network inference only.
* The hyperparameters (hidden sizes 128/256, batch 32) follow the paper; only the learning rate was
  changed after observing the lr 1e-4 results, and it was changed for all models equally.

## Reference

Kangrui Xue and Jane Wu. *Data-Driven Clothing for Interactive Applications.* Stanford CS 229 project report.
