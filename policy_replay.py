"""Replay sequential query policies on real Halmstad per-crop votes.
Each clip is a track; crops are queried in frame order; the task is drone vs bird.
Policies (parameters fit on training folds, evaluated on held-out clips, 5-fold grouped):
  FIXED-k : majority of the first k votes
  IND     : sequential Bayes assuming i.i.d. votes given class; stop at posterior >= .95 or <= .05
  CORR    : same rule, but votes are beta-binomial given class (track-level random effect)
Reported: clip accuracy, mean queries per clip, mean stated confidence, overconfidence = confidence - accuracy."""
import numpy as np, pandas as pd, glob
from scipy.special import betaln
from scipy.optimize import minimize
rng = np.random.default_rng(0)

def load(f):
    d = pd.read_csv(f); d.columns = [c.strip() for c in d.columns]
    fr = d["file"].str.extract(r"__f(\d+)\.jpg")[0].astype(int); d = d.assign(fr=fr)
    return d.sort_values(["track", "fr"])

def bb_fit(K, N):
    """MLE of beta-binomial (a,b) from per-clip counts."""
    def nll(p):
        a, b = np.exp(p)
        return -np.sum(betaln(K + a, N - K + b) - betaln(a, b))
    m = (K.sum() + .5) / (N.sum() + 1); r = minimize(nll, np.log([m * 5, (1 - m) * 5]), method="Nelder-Mead")
    return np.exp(r.x)

def run(d, thr=.95):
    rng = np.random.default_rng(0)
    clips = d.groupby("track").agg(truth=("truth", "first"), votes=("vote", list)).reset_index()
    clips = clips[clips.votes.map(len) >= 3].reset_index(drop=True)
    folds = rng.permutation(len(clips)) % 5; out = []
    for f in range(5):
        tr, te = clips[folds != f], clips[folds == f]
        prior = tr.truth.mean(); par = {}
        for c in (0, 1):
            g = tr[tr.truth == c]; K = g.votes.map(sum).values; N = g.votes.map(len).values
            par[c] = dict(p=(K.sum() + .5) / (N.sum() + 1), ab=bb_fit(K, N))
        for _, r in te.iterrows():
            v = r.votes; rec = dict(truth=r.truth, n=len(v))
            for k in (1, 3, 5):
                s = v[:k]; rec[f"fix{k}"] = int(np.mean(s) > .5) if np.mean(s) != .5 else int(prior > .5)
            for name in ("IND", "CORR"):
                for t in range(1, len(v) + 1):
                    K, N = sum(v[:t]), t
                    if name == "IND":
                        l = {c: K * np.log(par[c]["p"]) + (N - K) * np.log(1 - par[c]["p"]) for c in (0, 1)}
                    else:
                        l = {c: betaln(K + par[c]["ab"][0], N - K + par[c]["ab"][1]) - betaln(*par[c]["ab"]) for c in (0, 1)}
                    post = 1 / (1 + np.exp(l[0] - l[1]) * (1 - prior) / prior)
                    if post >= thr or post <= 1 - thr: break
                rec[name] = int(post > .5); rec[name + "_q"] = t; rec[name + "_conf"] = max(post, 1 - post)
            out.append(rec)
    return pd.DataFrame(out)

rows = []
for f in sorted(glob.glob("../results/*_*.csv")):
    if "tau_ml" in f or "scenes" in f: continue
    name = f.split("/")[-1][:-4]; R = run(load(f))
    row = dict(verifier=name, clips=len(R), crops_avail=R.n.mean())
    for k in (1, 3, 5): row[f"fix{k}_acc"] = (R[f"fix{k}"] == R.truth).mean()
    for p in ("IND", "CORR"):
        acc = (R[p] == R.truth).mean(); conf = R[p + "_conf"].mean()
        row.update({f"{p}_acc": acc, f"{p}_q": R[p + "_q"].mean(), f"{p}_conf": conf, f"{p}_over": conf - acc})
    rows.append(row)
T = pd.DataFrame(rows); pd.set_option("display.width", 250); print(T.round(3).to_string(index=False)); T.to_csv("policy_replay.csv", index=False)

# matched-budget comparison: sweep stopping thresholds
print("\nthreshold sweep: (mean queries, accuracy, overconfidence)")
sweep = []
for f in sorted(glob.glob("../results/*_*.csv")):
    if "tau_ml" in f or "scenes" in f or "qwen2" in f: continue
    name = f.split("/")[-1][:-4]; d = load(f)
    for thr in (.8, .9, .95, .99, .999):
        R = run(d, thr)
        for p in ("IND", "CORR"):
            sweep.append(dict(verifier=name, thr=thr, policy=p, q=R[p + "_q"].mean(), acc=(R[p] == R.truth).mean(),
                              over=R[p + "_conf"].mean() - (R[p] == R.truth).mean()))
S = pd.DataFrame(sweep); S.to_csv("policy_sweep.csv", index=False)
for v, g in S.groupby("verifier"):
    print(v, "  ".join(f"{r.policy}@{r.thr}:({r.q:.1f},{r.acc:.3f},{r.over:+.3f})" for r in g.itertuples()))
