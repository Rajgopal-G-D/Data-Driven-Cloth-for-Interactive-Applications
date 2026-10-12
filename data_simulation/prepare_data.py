"""Step 2: generate dataset, inspect distances, clean outliers, split, save."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cloth_pipeline import make_dataset, clean

os.makedirs("data", exist_ok=True)
os.makedirs("figures", exist_ok=True)

X, Y = make_dataset(n=10000, joints=14, verts=2170)
print(f"Raw dataset: X {X.shape}, Y {Y.shape}")

# Mean squared L2 distance of offsets per example (as in the paper)
dist = (Y.reshape(len(Y), -1, 3) ** 2).sum(-1).mean(1)
print(f"Distance: min {dist.min():.4f}  mean {dist.mean():.4f}  max {dist.max():.4f}")

Xc, Yc = clean(X, Y, keep_quantile=0.98)
thr = np.quantile(dist, 0.98)
print(f"Removed {len(X) - len(Xc)} outliers above threshold {thr:.4f}; {len(Xc)} remain")

n_train, n_val = 6000, 2000
print(f"Split: train {n_train}, val {n_val}, test {len(Xc) - n_train - n_val}")

plt.figure(figsize=(5, 3.5))
plt.hist(dist, bins=30)
plt.axvline(thr, color="r", ls="--", label=f"threshold = {thr:.3f}")
plt.xlabel("Mean squared offset distance"); plt.ylabel("Examples")
plt.title("Distribution of offset distances"); plt.legend(); plt.tight_layout()
plt.savefig("figures/distance_histogram.png", dpi=150)

np.savez("data/dataset.npz", X=Xc, Y=Yc)
print("Saved data/dataset.npz and figures/distance_histogram.png")