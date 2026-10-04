"""Step 3: PCA on training offsets -> explained variance + reconstruction error."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("figures", exist_ok=True)
N_TRAIN, K_MAX = 6000, 150

d = np.load("data/dataset.npz")
Y = d["Y"][:N_TRAIN].astype(np.float64)       # fit on TRAIN only
n, D = Y.shape
V = D // 3

mu = Y.mean(0)
Yc = Y - mu
_, S, Vt = np.linalg.svd(Yc, full_matrices=False)   # rows of Vt = principal components

explained = np.cumsum(S ** 2) / np.sum(S ** 2)
k90 = int(np.searchsorted(explained, 0.90)) + 1
k99 = int(np.searchsorted(explained, 0.99)) + 1
print(f"Components for 90% variance: {k90}")
print(f"Components for 99% variance: {k99}")

# Reconstruction error vs k. Components are orthonormal, so the residual energy of an
# example after keeping k components is ||x||^2 - sum_{l<=k} c_l^2. Dividing by the
# number of vertices gives the mean squared per-vertex error of that example.
C = Yc @ Vt[:K_MAX].T                                  # (n, K_MAX) coefficients
total = (Yc ** 2).sum(1)                               # (n,)
resid = total[:, None] - np.cumsum(C ** 2, axis=1)     # (n, K_MAX)
per_vertex_err = np.maximum(resid, 0) / V
avg_err = per_vertex_err.mean(0)
max_err = per_vertex_err.max(0)
ks = np.arange(1, K_MAX + 1)

fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
ax[0].plot(np.arange(1, K_MAX + 1), explained[:K_MAX], "r")
ax[0].axvline(k90, ls="--", c="k", lw=0.8, label=f"k={k90} (90%)")
if k99 <= K_MAX:
    ax[0].axvline(k99, ls="--", c="gray", lw=0.8, label=f"k={k99} (99%)")
ax[0].set(xlabel="Number of principal components k", ylabel="Cumulative explained variance",
          title="(a) Explained variance")
ax[0].legend()
ax[1].plot(ks, avg_err, "b", label="avg. (train)")
ax[1].plot(ks, max_err, "r", label="max. (train)")
ax[1].set(xlabel="Number of principal components k", ylabel="Squared per-vertex error",
          title="(b) Reconstruction error")
ax[1].legend()
plt.tight_layout()
plt.savefig("figures/pca_analysis.png", dpi=150)

json.dump(dict(k90=k90, k99=k99, explained=explained[:K_MAX].tolist(),
               avg_err=avg_err.tolist(), max_err=max_err.tolist()),
          open("pca_stats.json", "w"))
print("Saved figures/pca_analysis.png and pca_stats.json")
