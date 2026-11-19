"""
Uncertainty for tau, the clip-level SD of a model's error log-odds.

Replaces the variational fit (biased upward: about 0.3 when the truth is 0)
with exact maximum likelihood for the random-intercept logistic model
    logit P(err_ij) = X_ij b + tau * u_i,   u_i ~ N(0, 1),
integrating u by 30-point Gauss-Hermite quadrature (same model as lme4::glmer
with nAGQ > 1). Reports for each model x sensor x class:
  tau_hat, 95% profile-likelihood CI, 95% clip-bootstrap percentile CI,
  LR test of tau = 0 (boundary: p = 0.5 * P(chi2_1 > LR)), Holm-adjusted p.
Usage: python tau_ml.py file1.csv [file2.csv ...]   (needs err or vote+truth,
       track, size, dist columns; a crops_<s>/meta.json is used to add size/dist
       when missing)
"""
import sys, os, json, warnings
import numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, log_expit
from scipy.stats import chi2
warnings.filterwarnings("ignore")
XQ, WQ = np.polynomial.hermite.hermgauss(30)
UQ, LWQ = np.sqrt(2) * XQ, np.log(WQ / np.sqrt(np.pi))


def design(d, with_class=False):
    cols = [np.ones(len(d)), np.log(d["size"].values)]
    for lev in sorted(d["dist"].astype(str).unique())[1:]:
        cols.append((d["dist"].astype(str).values == lev).astype(float))
    if with_class:
        cols.append((d["truth"].values == 1).astype(float))
    return np.column_stack(cols)


def negll(b, tau, X, y, g, ng):
    eta = (X @ b)[:, None] + tau * UQ[None, :]                       # (n,Q)
    lp = np.where(y[:, None] == 1, log_expit(eta), log_expit(-eta))
    S = np.zeros((ng, len(UQ))); np.add.at(S, g, lp)
    S = S + LWQ[None, :]
    m = S.max(1, keepdims=True)
    return -float((m.squeeze() + np.log(np.exp(S - m).sum(1))).sum())


def fit_b(tau, X, y, g, ng, b0):
    r = minimize(negll, b0, args=(tau, X, y, g, ng), method="BFGS")
    return r.x, r.fun


def fit(X, y, g):
    ng = g.max() + 1; b0 = np.zeros(X.shape[1])
    best = None
    for t0 in (0.3, 1.0, 2.0):                                       # multi-start
        r = minimize(lambda p: negll(p[:-1], np.exp(p[-1]), X, y, g, ng),
                     np.r_[b0, np.log(t0)], method="BFGS")
        if best is None or r.fun < best.fun:
            best = r
    tau = float(np.exp(best.x[-1])); b = best.x[:-1]
    _, nll0 = fit_b(0.0, X, y, g, ng, b)                             # tau = 0 model
    lr = max(0.0, 2 * (nll0 - best.fun))
    return tau, b, best.fun, lr


def profile_ci(X, y, g, tau, b, nll, level=0.95):
    ng = g.max() + 1; cut = chi2.ppf(level, 1) / 2

    def gap(t):
        return fit_b(t, X, y, g, ng, b)[1] - nll - cut
    lo = 0.0
    if gap(1e-6) > 0:                                                # lower bound above 0
        a, c = 1e-6, tau
        for _ in range(25):
            m = (a + c) / 2; a, c = (m, c) if gap(m) > 0 else (a, m)
        lo = c
    a, c = tau, tau * 2 + 1
    while gap(c) < 0 and c < 20:
        c *= 1.5
    for _ in range(25):
        m = (a + c) / 2; a, c = (a, m) if gap(m) > 0 else (m, c)
    return lo, c


def boot_ci(d, with_class, B=200, seed=0):
    rng = np.random.default_rng(seed); clips = d["track"].unique(); out = []
    by = {c: d[d.track == c] for c in clips}
    for _ in range(B):
        pick = rng.choice(clips, len(clips), replace=True)
        bd = pd.concat([by[c].assign(track=f"{c}#{k}") for k, c in enumerate(pick)])
        if bd.err.nunique() < 2:
            continue
        X = design(bd, with_class); g = pd.factorize(bd.track)[0]
        out.append(fit(X, bd.err.values, g)[0])
    return np.quantile(out, [0.025, 0.975]) if out else (np.nan, np.nan)


def load(path):
    d = pd.read_csv(path)
    if "err" not in d:
        d["err"] = (d["vote"].astype(int) != d["truth"].astype(int)).astype(int)
    if "truth" in d and not pd.api.types.is_numeric_dtype(d["truth"]):
        d["truth"] = (d["truth"].astype(str) == "drone").astype(int)
    if "y" in d and "truth" not in d:
        d["truth"] = d["y"]
    if ("size" not in d or "dist" not in d) and "file" in d:
        s = "IR" if "_IR" in os.path.basename(path) else "V"
        meta = pd.DataFrame(json.load(open(f"crops_{s}/meta.json")))[["file", "size", "dist"]]
        d = d.drop(columns=[c for c in ("size", "dist") if c in d]).merge(meta, on="file")
    t = d.groupby("track").size()
    return d[d.track.isin(t[t >= 3].index)].copy()


def run(paths, B=200):
    rows = []
    for p in paths:
        d = load(p); name = os.path.basename(p).replace(".csv", "")
        for cls, sub, wc in [("drone", d[d.truth == 1], False), ("bird", d[d.truth == 0], False), ("pooled", d, True)]:
            if sub.err.nunique() < 2 or sub.err.mean() < 0.01:
                rows.append(dict(file=name, cls=cls, n_clips=sub.track.nunique(), err=sub.err.mean())); continue
            X = design(sub, wc); g = pd.factorize(sub.track)[0]; y = sub.err.values
            tau, b, nll, lr = fit(X, y, g)
            lo, hi = profile_ci(X, y, g, tau, b, nll)
            blo, bhi = boot_ci(sub, wc, B)
            p = 0.5 * chi2.sf(lr, 1) if lr > 0 else 0.5
            rows.append(dict(file=name, cls=cls, n_clips=sub.track.nunique(), err=round(sub.err.mean(), 3),
                             tau=round(tau, 2), prof_lo=round(lo, 2), prof_hi=round(hi, 2),
                             boot_lo=round(blo, 2), boot_hi=round(bhi, 2), LR=round(lr, 1), p=p))
            print(rows[-1], flush=True)
    R = pd.DataFrame(rows)
    m = R["cls"] != "pooled"
    if "p" in R:
        pc = R.loc[m & R.p.notna(), "p"].sort_values()
        k = len(pc); adj = {}
        run_max = 0
        for i, (idx, pv) in enumerate(pc.items()):
            run_max = max(run_max, min(1.0, (k - i) * pv)); adj[idx] = run_max
        R["p_holm"] = pd.Series(adj)
    return R


if __name__ == "__main__":
    B = int(os.environ.get("BOOT", "200"))
    R = run(sys.argv[1:], B)
    pd.set_option("display.width", 200)
    print(R.to_string(index=False))
    R.to_csv("tau_ml_results.csv", index=False)
