"""
Fixes for two weaknesses a statistician would raise.
 (2) Exchangeable correlation: measure how error dependence decays with time lag within a track, and fit
     phi(k) = a + b * exp(-k / L)   (random-intercept plateau a  +  serial component b decaying over L frames).
 (1) The design-effect correction, derived. If frame log-likelihood ratios l_j are Gaussian with class means mu_c,
     common variance s^2 and within-track correlation matrix R, the exact track-level LLR is
         LLR = (mu1 - mu0)/s^2 * 1' R^{-1} (l - m 1),   m = (mu0 + mu1)/2.
     With exchangeable R this is (mu1 - mu0)/s^2 * (sum l - n m) / (1 + (n-1) rho), and for calibrated
     frame LLRs (mu1 = s^2/2 = -mu0) it reduces exactly to  sum(l) / (1 + (n-1) rho)  -- the design-effect rule.
     We compare: independent, mean, design effect (ANOVA rho), derived exchangeable (estimated mu, s),
     and derived GLS with random intercept + AR(1) correlation  R_jk = a + b*phi^|j-k| (j != k).
Usage (Colab, CPU is fine):  %run /content/drive/MyDrive/counter_uas_attack/fix12.py
Or locally:  python fix12.py scores.csv out.txt
"""
import sys, numpy as np, pandas as pd
from scipy.optimize import curve_fit
from sklearn.linear_model import LogisticRegression

WORK = "/content/drive/MyDrive/counter_uas_attack"
SC = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1].endswith(".csv") else f"{WORK}/avid_b_scores.csv"
OUT = sys.argv[2] if len(sys.argv) > 2 else f"{WORK}/fix12_results.txt"
SEG, THR = 25, 0.25
lines = []
def say(x):
    print(x, flush=True); lines.append(str(x)); open(OUT, "w").write("\n".join(lines))
lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))

D = pd.read_csv(SC).dropna(subset=["ir", "rgb"]).sort_values(["video", "frame"]).reset_index(drop=True)
D["seg"] = D.video + "_" + (D.frame // SEG).astype(str)

# ---------------------------------------------------------------- (2) lag decay of error dependence
LAGS = np.array([1, 2, 3, 5, 8, 12, 18, 25, 40, 60, 90, 130, 180])
decay = lambda k, a, b, L: a + b * np.exp(-k / L)
say("=== (2) How error dependence decays with time lag (AVID, frame rate 25/s) ===")
for sen in ("ir", "rgb"):
    for cls, nm in ((1, "misses on drone frames"), (0, "false alarms on bird frames")):
        g = D[D.truth == cls]; phis = []
        for k in LAGS:
            x, y = [], []
            for _, h in g.groupby("video"):
                e = (((h[sen] >= THR).astype(int)) != cls).values; fr = h.frame.values
                pos = {f: i for i, f in enumerate(fr)}
                for i, f in enumerate(fr):
                    j = pos.get(f + k)
                    if j is not None: x.append(e[i]); y.append(e[j])
            x, y = np.array(x), np.array(y)
            phis.append(np.corrcoef(x, y)[0, 1] if len(x) > 30 and x.std() > 0 and y.std() > 0 else np.nan)
        phis = np.array(phis); ok = ~np.isnan(phis)
        try:
            (a, b, L), _ = curve_fit(decay, LAGS[ok], phis[ok], p0=[.3, .4, 10], bounds=([0, 0, .3], [1, 1, 500]))
            say(f"{sen} {nm}: error correlation at lag 1 = {phis[0]:.2f}, lag 25 = {phis[list(LAGS).index(25)]:.2f}, lag 180 = {phis[-1]:.2f} | "
                f"fit: plateau a = {a:.2f} (lasting track effect), serial b = {b:.2f}, decay length L = {L:.1f} frames ({L / 25:.2f} s) | "
                f"share of lag-1 dependence that persists = {a / (a + b):.2f}")
        except Exception as ex:
            say(f"{sen} {nm}: fit failed ({ex}); correlations {np.round(phis, 2).tolist()}")

# ---------------------------------------------------------------- (1)+(2) derived fusion rules
def metrics(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6); bins = np.minimum((p * 10).astype(int), 9)
    ece = sum(abs(y[bins == k].mean() - p[bins == k].mean()) * (bins == k).mean() for k in range(10) if (bins == k).any())
    return dict(acc=((p > .5) == y).mean(), brier=((p - y) ** 2).mean(), nll=-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean(),
                ece=ece, over=np.maximum(p, 1 - p).mean() - ((p > .5) == y).mean())

D["zI"], D["zV"] = lg(D.ir), lg(D.rgb)
rules = ["independent over time", "segment mean", "design effect (ANOVA rho)", "derived, exchangeable", "derived, random intercept + AR(1)"]
out = {r: [] for r in rules}; fitted = []
for v in D.video.unique():
    tr, te = D[D.video != v], D[D.video == v]
    m = LogisticRegression(C=1e3, max_iter=2000).fit(tr[["zI", "zV"]].values, tr.truth.values)
    prior = tr.truth.mean(); lp = np.log(prior / (1 - prior))
    ltr = lg(m.predict_proba(tr[["zI", "zV"]].values)[:, 1]) - lp; lte = lg(m.predict_proba(te[["zI", "zV"]].values)[:, 1]) - lp
    T = pd.DataFrame(dict(seg=tr.seg.values, y=tr.truth.values, l=ltr))
    mu = {c: T[T.y == c].l.mean() for c in (0, 1)}; mbar = (mu[0] + mu[1]) / 2
    resid = T.l - T.y.map(mu); s2 = resid.var()
    # ANOVA ICC (as in the original design-effect rule)
    iccs = []
    for c in (0, 1):
        h = T[T.y == c]; ms = h.groupby("seg").l.mean(); ns = h.groupby("seg").size(); k = ns.mean()
        msb = (ns * (ms - h.l.mean()) ** 2).sum() / max(1, len(ms) - 1); msw = ((h.l - h.seg.map(ms)) ** 2).sum() / max(1, len(h) - len(ms))
        iccs.append(max(0.0, (msb - msw) / (msb + (k - 1) * msw)))
    rho = float(np.mean(iccs))
    # within-segment autocorrelation of residual evidence, fitted as a + b*phi^k
    T["r"] = resid.values; ks = np.arange(1, SEG); ac = []
    for k in ks:
        num = den = 0.0
        for _, h in T.groupby("seg"):
            r = h.r.values
            if len(r) > k: num += (r[:-k] * r[k:]).sum(); den += len(r) - k
        ac.append(num / den / s2 if den else np.nan)
    ac = np.array(ac)
    try:
        (a, b, ph), _ = curve_fit(lambda k, a, b, ph: a + b * ph ** k, ks, ac, p0=[.4, .3, .7], bounds=([0, 0, 0], [.999, .999, .999]))
        if a + b > .999: b = .999 - a
    except Exception:
        a, b, ph = rho, 0.0, 0.0
    fitted.append((rho, a, b, ph, mu[0], mu[1], s2))
    for s, h in pd.DataFrame(dict(seg=te.seg.values, y=te.truth.values, l=lte)).groupby("seg", sort=False):
        l = h.l.values; n = len(l); y = h.y.iloc[0]
        J = np.arange(n); R = a + b * ph ** np.abs(J[:, None] - J[None, :]); np.fill_diagonal(R, 1.0)
        w = np.linalg.solve(R, np.ones(n))
        rho_ex = (R.sum() - n) / (n * (n - 1)) if n > 1 else 0.0          # average correlation implied by the fitted R
        L = {"independent over time": l.sum(), "segment mean": l.mean(),
             "design effect (ANOVA rho)": l.sum() / (1 + (n - 1) * rho),
             "derived, exchangeable": (mu[1] - mu[0]) / s2 * (l.sum() - n * mbar) / (1 + (n - 1) * rho_ex),
             "derived, random intercept + AR(1)": (mu[1] - mu[0]) / s2 * float(w @ (l - mbar))}
        for r_, val in L.items(): out[r_].append((y, 1 / (1 + np.exp(-np.clip(lp + val, -50, 50)))))
F = np.array(fitted).mean(0)
say("\n=== (1)+(2) Track-level fusion over 25-frame segments, leave-one-video-out ===")
say(f"fitted (mean over folds): ANOVA rho {F[0]:.2f} | residual-evidence correlation a + b*phi^k with a = {F[1]:.2f}, b = {F[2]:.2f}, phi = {F[3]:.2f} | "
    f"class means of frame LLR {F[4]:.2f}, {F[5]:.2f} (calibrated would be {-F[6] / 2:.2f}, {F[6] / 2:.2f}), residual variance {F[6]:.2f}")
for r_, L_ in out.items():
    y, p = map(np.array, zip(*L_)); mm = metrics(y, p)
    say(f"  {r_}: acc {mm['acc']:.3f}, Brier {mm['brier']:.4f}, NLL {mm['nll']:.3f}, ECE {mm['ece']:.3f}, overconfidence {mm['over']:+.3f}")
say("[END] fix12 finished")
