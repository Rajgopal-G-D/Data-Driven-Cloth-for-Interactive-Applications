"""Step 4: train linear baseline + MLP variants (Table 1 ablation) on data/dataset.npz."""
import argparse, json, os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cloth_pipeline import DEV, split, fit_pca, build_model, train, mse, time_inference

ap = argparse.ArgumentParser()
ap.add_argument("--epochs", type=int, default=200)   # paper: 1000
ap.add_argument("--lr", type=float, default=1e-4)
a = ap.parse_args()

torch.manual_seed(0)
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("figures", exist_ok=True)

d = np.load("data/dataset.npz")
train_set, val_set, test_set = split(d["X"], d["Y"])
d_in, d_out = d["X"].shape[1], d["Y"].shape[1]
print(f"device={DEV}  in={d_in} out={d_out}  "
      f"train/val/test={len(train_set[0])}/{len(val_set[0])}/{len(test_set[0])}")

mu, Vt, _ = fit_pca(train_set[1])      # PCA fitted on training data only

configs = [
    ("0_linear",       [],         None),
    ("1_offsets",      [128, 256], None),
    ("2_pca256",       [128, 256], 256),
    ("3_pca128",       [128, 256], 128),
    ("4_pca64",        [128, 256], 64),
]

results, curves = [], {}
for name, hidden, k in configs:
    print(f"\n=== {name} ===")
    model = build_model(d_in, d_out, hidden, (mu, Vt) if k else None, k)
    model, hist = train(model, train_set, val_set, a.epochs, a.lr)
    torch.save(model.state_dict(), f"checkpoints/{name}.pt")
    curves[name] = hist
    results.append(dict(model=name, hidden=hidden, pca_k=k,
                        train=mse(model, *train_set), val=mse(model, *val_set),
                        test=mse(model, *test_set),
                        ms_per_mesh=time_inference(model, d_in) * 1e3))

print("\n{:<12}{:>14}{:>14}{:>14}{:>12}".format("Model", "Train MSE", "Val MSE", "Test MSE", "ms/mesh"))
for r in results:
    print("{:<12}{:>14.3e}{:>14.3e}{:>14.3e}{:>12.4f}".format(
        r["model"], r["train"], r["val"], r["test"], r["ms_per_mesh"]))
best = min(results, key=lambda r: r["val"])
print(f"\nBest by validation MSE: {best['model']}")

json.dump(dict(epochs=a.epochs, lr=a.lr, results=results, best=best["model"],
               curves={k: v for k, v in curves.items()}),
          open("results.json", "w"))

plt.figure(figsize=(7, 4))
for name, hist in curves.items():
    plt.plot([h[0] for h in hist], [h[2] for h in hist], label=name)
plt.yscale("log"); plt.xlabel("Epoch"); plt.ylabel("Validation MSE")
plt.title("Validation loss per model"); plt.legend(); plt.tight_layout()
plt.savefig("figures/training_curves.png", dpi=150)
print("Saved results.json, checkpoints/, figures/training_curves.png")
