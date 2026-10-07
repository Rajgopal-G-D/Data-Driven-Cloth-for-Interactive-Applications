import json
R = json.load(open("results.json"))
for r in R["results"]:
    print(r["model"], f"train {r['train']:.2e}  val {r['val']:.2e}  test {r['test']:.2e}")
P = json.load(open("pca_stats.json"))
for k in (64, 128, 256):
    if k <= len(P["avg_err"]): print("PCA floor k =", k, P["avg_err"][k-1])