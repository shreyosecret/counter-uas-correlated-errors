import numpy as np, pandas as pd, sys
sys.path.insert(0, "../antiuav"); import attack as A
from scipy.stats import chi2
S = pd.read_csv("../results/halmstad_scenes.csv")
def load(f):
    d = pd.read_csv(f); d.columns = [c.strip() for c in d.columns]; return d
meta = pd.concat([load("../results/clip_V.csv"), load("../results/clip_IR.csv")])[["file", "size", "dist"]]
rows = []
for name in ("clip_V", "clip_IR", "qwen2_V", "probe_V", "probe_IR"):
    d = load(f"../results/{name}.csv")
    if "size" not in d or "dist" not in d: d = d.drop(columns=[c for c in ("size", "dist") if c in d]).merge(meta, on="file")
    d["err"] = (d.vote != d.truth).astype(int)
    for cls, lab in ((1, "drone"), (0, "bird")):
        g = d[d.truth == cls].merge(S, on="track")
        t = g.groupby("track").size(); g = g[g.track.isin(t[t >= 3].index)]
        if g.err.mean() < .02 or g.err.mean() > .98: rows.append(dict(model=name, cls=lab, note="too few errors")); continue
        X = np.column_stack([np.ones(len(g)), np.log(g["size"].values)] + [(g.dist.astype(str) == l).astype(float).values for l in sorted(g.dist.astype(str).unique())[1:]])
        clip = pd.factorize(g.track)[0]
        for K in ("K20",):
            sc = pd.factorize(g.groupby("track")[K].first().reindex(g.track).values)[0]
            F = A.fit2(X, g.err.values, clip, sc); NT = A.fit2(X, g.err.values, clip, sc, fix_tau0=True)
            share = F["tau"] ** 2 / (F["tau"] ** 2 + F["sig"] ** 2 + 1e-12)
            rows.append(dict(model=name, cls=lab, K=K, scenes=sc.max() + 1, clips=clip.max() + 1, err=round(g.err.mean(), 3),
                             tau=round(F["tau"], 2), sigma=round(F["sig"], 2), clip_share=round(share, 2),
                             p_tau=0.5 * chi2.sf(max(0, 2 * (NT["nll"] - F["nll"])), 1))); print(rows[-1], flush=True)
R = pd.DataFrame(rows); pd.set_option("display.width", 200); print(R.to_string(index=False)); R.to_csv("scene_split_results.csv", index=False)
