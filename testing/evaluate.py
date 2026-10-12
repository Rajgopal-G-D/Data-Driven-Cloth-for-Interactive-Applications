"""Step 5: final evaluation on the TEST set: tables, timing, error analysis."""
import json, os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cloth_pipeline import DEV, split, build_model, time_inference

os.makedirs("figures", exist_ok=True)
R = json.load(open("results.json"))
d = np.load("data/dataset.npz")
_, _, (Xte, Yte) = split(d["X"], d["Y"])
d_in, d_out = Xte.shape[1], Yte.shape[1]
V = d_out // 3
print(f"device={DEV}  test examples={len(Xte)}  vertices={V}")

rows, vert_err, ex_err = [], {}, {}
for r in R["results"]:
    name, k = r["model"], r["pca_k"]
    dummy = (torch.zeros(d_out, device=DEV), torch.zeros(k, d_out, device=DEV)) if k else None
    model = build_model(d_in, d_out, r["hidden"], dummy, k)
    model.load_state_dict(torch.load(f"checkpoints/{name}.pt", map_location=DEV))
    model.eval()
    with torch.no_grad():
        pred = model(Xte)
    sq = ((pred - Yte) ** 2).reshape(len(Xte), V, 3).sum(-1)        # (n, V) squared per-vertex error
    vert_err[name] = sq.mean(0).cpu().numpy()                       # avg over examples
    ex_err[name] = sq.mean(1).cpu().numpy()                         # avg over vertices
    rel = (((pred - Yte) ** 2).sum() / (Yte ** 2).sum()).item() * 100
    ms1 = time_inference(model, d_in, runs=500) * 1e3
    rows.append(dict(name=name, test_mse=((pred - Yte) ** 2).mean().item(),
                     vertex_err=float(ex_err[name].mean()), rel=rel,
                     worst=float(ex_err[name].max()), best=float(ex_err[name].min()),
                     ms=ms1, params=sum(p.numel() for p in model.parameters() if p.requires_grad)))

best = R["best"]
best_row = next(x for x in rows if x["name"] == best)
base = next(x for x in rows if x["name"] == "0_linear")

lines = ["## Test-set results", "",
         "| Model | Test MSE | Mean sq. per-vertex err | Relative err (%) | Best ex. | Worst ex. | Trainable params | Inference (ms/mesh) |",
         "|---|---|---|---|---|---|---|---|"]
for x in rows:
    star = " (best on val)" if x["name"] == best else ""
    lines.append(f"| {x['name']}{star} | {x['test_mse']:.3e} | {x['vertex_err']:.3e} | {x['rel']:.2f} | "
                 f"{x['best']:.2e} | {x['worst']:.2e} | {x['params']:,} | {x['ms']:.4f} |")
lines += ["", f"Best model (by validation MSE): **{best}**; improvement over linear baseline in test MSE: "
          f"**{(1 - best_row['test_mse'] / base['test_mse']) * 100:.1f}%**.",
          f"Throughput of best model at batch size 1: ~{1000 / best_row['ms']:.0f} meshes/s on {DEV}."]
open("results_table.md", "w").write("\n".join(lines))
print("\n".join(lines))

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
names = [x["name"] for x in rows]
ax[0].bar(names, [x["test_mse"] for x in rows]); ax[0].set_yscale("log")
ax[0].set(title="(a) Test MSE by model", ylabel="MSE"); ax[0].tick_params(axis="x", rotation=30)
for n in ("0_linear", best):
    ax[1].plot(np.sort(np.sqrt(vert_err[n]))[::-1], label=n)
ax[1].set(title="(b) Per-vertex RMS error (sorted)", xlabel="Vertex rank", ylabel="RMS error"); ax[1].legend()
ax[2].hist(ex_err[best], bins=40)
ax[2].set(title=f"(c) Per-example error, {best}", xlabel="Mean sq. per-vertex error", ylabel="Test examples")
plt.tight_layout(); plt.savefig("figures/evaluation.png", dpi=150)

# Which test poses are easiest / hardest for the best model
order = np.argsort(ex_err[best])
np.save("figures/best_worst_indices.npy", np.array([order[0], order[-1]]))
print(f"\nLowest-error test example idx {order[0]} ({ex_err[best][order[0]]:.3e}); "
      f"highest idx {order[-1]} ({ex_err[best][order[-1]]:.3e})")
print("Saved results_table.md, figures/evaluation.png")
