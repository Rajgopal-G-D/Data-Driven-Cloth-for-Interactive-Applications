"""
Data-driven clothing mini-project (after Xue & Wu, "Data-Driven Clothing for
Interactive Applications").

Pipeline:
  1. Build dataset: input = 14 joint quaternions (56-d), output = vertex offsets (V*3)
  2. Clean + split (6000 / 2000 / rest)
  3. PCA on training offsets (via SVD)
  4. Models: linear regression, MLP -> offsets, MLP -> PCA coeffs -> fixed PC layer
  5. Ablation over number of PCA coefficients, MSE loss, Adam, batch 32
  6. Inference timing

NOTE: the paper's data comes from Blender skinning + PhysBAM simulation, which is
not available. make_dataset() generates a synthetic stand-in with the same shapes.
Replace it with your own loader (see load_dataset) if you get real data.
"""
import argparse, json, time
import numpy as np
import torch
import torch.nn as nn

DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------- 1. data
def random_quats(n, joints, rng):
    q = rng.normal(size=(n, joints, 4))
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    q *= np.sign(q[..., :1] + 1e-9)  # keep w >= 0 (q and -q are the same rotation)
    return q.reshape(n, joints * 4).astype(np.float32)


def make_dataset(n=10000, joints=14, verts=400, seed=0):
    """Synthetic 'cloth': linear (skinning-like) part + smooth nonlinear part + noise."""
    rng = np.random.default_rng(seed)
    X = random_quats(n, joints, rng)
    d_in, d_out = joints * 4, verts * 3
    W_lin = rng.normal(size=(d_in, d_out)) * 0.02
    H = rng.normal(size=(d_in, 64))
    W_nl = rng.normal(size=(64, d_out)) * 0.02
    Y = X @ W_lin + np.tanh(2.0 * (X @ H)) @ W_nl
    Y += rng.normal(size=Y.shape) * 1e-3
    return X, Y.astype(np.float32)


def load_dataset(path=None, **kw):
    # TODO: if you obtain real data, load X (N,56) and Y (N,V*3) here.
    return make_dataset(**kw)


def clean(X, Y, keep_quantile=0.98):
    """Mimic the paper's distance-threshold filtering of faulty simulations."""
    dist = (Y.reshape(len(Y), -1, 3) ** 2).sum(-1).mean(1)
    mask = dist <= np.quantile(dist, keep_quantile)
    return X[mask], Y[mask]


def split(X, Y, n_train=6000, n_val=2000):
    t = lambda a: torch.tensor(a, device=DEV)
    return [(t(X[a:b]), t(Y[a:b])) for a, b in
            [(0, n_train), (n_train, n_train + n_val), (n_train + n_val, len(X))]]


# ---------------------------------------------------------------- 3. PCA
def fit_pca(Ytr):
    mu = Ytr.mean(0)
    _, S, Vt = torch.linalg.svd(Ytr - mu, full_matrices=False)
    explained = (S ** 2).cumsum(0) / (S ** 2).sum()
    return mu, Vt, explained  # Vt rows = principal components u_l


# ---------------------------------------------------------------- 4. models
def build_model(d_in, d_out, hidden, pca=None, k=None):
    layers, prev = [], d_in
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.ReLU()]
        prev = h
    if pca is None:
        layers.append(nn.Linear(prev, d_out))          # direct offsets
    else:
        mu, Vt = pca
        layers.append(nn.Linear(prev, k))               # PCA coefficients c_l
        pc = nn.Linear(k, d_out)                        # fixed "PC layer": sum c_l u_l + mu
        pc.weight.data = Vt[:k].T.contiguous().clone()
        pc.bias.data = mu.clone()
        for p in pc.parameters():
            p.requires_grad = False
        layers.append(pc)
    return nn.Sequential(*layers).to(DEV)


def mse(model, X, Y):
    with torch.no_grad():
        return ((model(X) - Y) ** 2).mean().item()


def train(model, train_set, val_set, epochs, lr=1e-4, bs=32):
    Xtr, Ytr = train_set
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    best, best_state = float("inf"), None
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr), device=DEV)
        for i in range(0, len(Xtr), bs):
            idx = perm[i:i + bs]
            loss = ((model(Xtr[idx]) - Ytr[idx]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        v = mse(model, *val_set)
        if v < best:
            best = v
            best_state = {k_: t.clone() for k_, t in model.state_dict().items()}
        if (ep + 1) % max(1, epochs // 10) == 0:
            print(f"    epoch {ep+1:4d}  train {mse(model, Xtr, Ytr):.3e}  val {v:.3e}")
    model.load_state_dict(best_state)
    return model


def time_inference(model, d_in, runs=200):
    model.eval()
    x = torch.randn(1, d_in, device=DEV)
    with torch.no_grad():
        for _ in range(20):
            model(x)
        if DEV == "cuda": torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(runs):
            model(x)
        if DEV == "cuda": torch.cuda.synchronize()
    return (time.perf_counter() - t0) / runs


# ---------------------------------------------------------------- 5. main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=200)   # paper uses 1000
    ap.add_argument("--verts", type=int, default=400)    # paper uses 2170
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--out", default="results.json")
    a = ap.parse_args()
    torch.manual_seed(0)

    X, Y = load_dataset(verts=a.verts)
    X, Y = clean(X, Y)
    train_set, val_set, test_set = split(X, Y)
    d_in, d_out = X.shape[1], Y.shape[1]
    print(f"device={DEV}  train/val/test = {len(train_set[0])}/{len(val_set[0])}/{len(test_set[0])}")

    mu, Vt, explained = fit_pca(train_set[1])
    for target in (0.9, 0.99):
        print(f"PCs for {target:.0%} variance: {int((explained < target).sum()) + 1}")

    max_k = len(Vt)
    configs = [
        ("0 linear regression", [], None),
        ("1 relu_128_256 -> offsets", [128, 256], None),
        ("2 relu_128_256 -> PCA 256", [128, 256], min(256, max_k)),
        ("3 relu_128_256 -> PCA 128", [128, 256], 128),
        ("4 relu_128_256 -> PCA 64", [128, 256], 64),
    ]
    results = []
    for name, hidden, k in configs:
        print(f"\n[{name}]")
        model = build_model(d_in, d_out, hidden, (mu, Vt) if k else None, k)
        model = train(model, train_set, val_set, a.epochs, a.lr)
        r = dict(model=name, train=mse(model, *train_set), test=mse(model, *test_set),
                 ms_per_mesh=time_inference(model, d_in) * 1e3)
        results.append(r)

    print("\n{:<30}{:>14}{:>14}{:>14}".format("Model", "Train MSE", "Test MSE", "ms/mesh"))
    for r in results:
        print("{:<30}{:>14.3e}{:>14.3e}{:>14.4f}".format(r["model"], r["train"], r["test"], r["ms_per_mesh"]))
    json.dump(dict(results=results, explained_variance=explained.cpu().tolist()),
              open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
