"""
Checks requested in peer review (run in Colab; CPU is fine):
  (1) 37-session nested model with the cross-fitting fold as a covariate (fold/model confound).
  (2) Session-cluster bootstrap 95% CI for the cross-sensor Mantel-Haenszel odds ratio (315 sequences).
  (3) AVID: fusion on COMPLETE 25-frame segments only, with video-level bootstrap intervals.
  (4) AVID: video-level bootstrap intervals for the Dawid-Skene audit errors.
    from google.colab import drive; drive.mount('/content/drive')
    %run /content/drive/MyDrive/counter_uas_attack/dce_checks.py
"""
import os, sys, ast, numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.stats import chi2
WORK = sys.argv[1] if len(sys.argv) > 1 else "/content/drive/MyDrive/counter_uas_attack"
sys.path.insert(0, WORK); import attack as A
from statsmodels.stats.contingency_tables import StratifiedTable
from sklearn.linear_model import LogisticRegression
OUT = f"{WORK}/dce_checks_results.txt"; lines = []
def say(x):
    print(x, flush=True); lines.append(str(x)); open(OUT, "w").write("\n".join(lines))
rng = np.random.default_rng(2026)

# reuse the exact profile-likelihood code from sessions2.py
src = open(f"{WORK}/sessions2.py").read(); g = dict(np=np, pd=pd, A=A, minimize=minimize, chi2=chi2)
for n in ast.parse(src).body:
    if isinstance(n, ast.FunctionDef) and n.name in ("_fit", "profile_ci"): exec(ast.get_source_segment(src, n), g)
_fit, profile_ci = g["_fit"], g["profile_ci"]

V = pd.read_csv(f"{WORK}/xfit_votes.csv", dtype={"track": str, "session": str})

# ------------------------------------------------------------ (2) session-cluster bootstrap for the MH odds ratio
P = V.pivot_table(index=["track", "frame"], columns="sensor", values="err").dropna().astype(int)
tabs = {}
for tr, h in P.groupby(level=0):
    tb = np.array([[((h.infrared == 1) & (h.visible == 1)).sum(), ((h.infrared == 1) & (h.visible == 0)).sum()],
                   [((h.infrared == 0) & (h.visible == 1)).sum(), ((h.infrared == 0) & (h.visible == 0)).sum()]], float)
    if tb.sum() >= 2: tabs[tr] = tb
sess_of = V.groupby("track").session.first(); by_sess = {}
for tr in tabs: by_sess.setdefault(sess_of[tr], []).append(tr)
keys = sorted(by_sess); est = StratifiedTable(list(tabs.values())).oddsratio_pooled; boots = []
for _ in range(2000):
    pick = rng.choice(keys, len(keys), replace=True); T = [tabs[t] for k in pick for t in by_sess[k]]
    try:
        o = StratifiedTable(T).oddsratio_pooled
        if np.isfinite(o): boots.append(o)
    except Exception: pass
lo, hi = np.quantile(boots, [.025, .975])
say(f"\n=== (2) cross-sensor MH odds ratio, all sessions: {est:.2f}, session-cluster bootstrap 95% CI [{lo:.2f}, {hi:.2f}] "
    f"({len(tabs)} sequences, {len(keys)} sessions, {len(boots)} valid resamples)")

# ------------------------------------------------------------ AVID: shared pieces
D = pd.read_csv(f"{WORK}/avid_b_scores.csv").dropna(subset=["ir", "rgb"]).sort_values(["video", "frame"]).reset_index(drop=True)
D["pos"] = D.groupby("video").cumcount(); D["blk"] = D.pos // 25
D["seg"] = D.video + "_" + D.blk.astype(str)
full = D.groupby("seg").size(); D["complete"] = D.seg.map(full) == 25
old = (D.video + "_" + (D.frame // 25).astype(str)).nunique()
say(f"\n=== AVID segments: old definition (frame // 25) gave {old} segments, including short leftovers; "
    f"complete 25-frame blocks: {int((full == 25).sum())} (frames {int(D.complete.sum())} of {len(D)}) ===")
lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
D["zI"], D["zV"] = lg(D.ir), lg(D.rgb)

# ------------------------------------------------------------ (3) fusion on complete segments, video bootstrap
C = D[D.complete].copy(); rows = []
for v in C.video.unique():
    tr, te = C[C.video != v], C[C.video == v]
    m = LogisticRegression(C=1e3, max_iter=2000).fit(tr[["zI", "zV"]].values, tr.truth.values)
    prior = tr.truth.mean(); lp = np.log(prior / (1 - prior))
    ltr = lg(m.predict_proba(tr[["zI", "zV"]].values)[:, 1]) - lp; lte = lg(m.predict_proba(te[["zI", "zV"]].values)[:, 1]) - lp
    T = pd.DataFrame(dict(seg=tr.seg.values, y=tr.truth.values, l=ltr)); iccs = []
    for c in (0, 1):
        h = T[T.y == c]; ms = h.groupby("seg").l.mean(); k = 25
        msb = (k * (ms - h.l.mean()) ** 2).sum() / max(1, len(ms) - 1); msw = ((h.l - h.seg.map(ms)) ** 2).sum() / max(1, len(h) - len(ms))
        iccs.append(max(0.0, (msb - msw) / (msb + (k - 1) * msw)))
    rho = float(np.mean(iccs))
    for s, h in pd.DataFrame(dict(seg=te.seg.values, y=te.truth.values, l=lte)).groupby("seg"):
        l = h.l.values; n = len(l)
        for rule, L in (("independent over time", l.sum()), ("design effect", l.sum() / (1 + (n - 1) * rho)), ("segment mean", l.mean())):
            rows.append(dict(video=v, seg=s, y=h.y.iloc[0], rule=rule, p=1 / (1 + np.exp(-np.clip(lp + L, -50, 50)))))
R = pd.DataFrame(rows)
def met(d):
    p = np.clip(d.p.values, 1e-6, 1 - 1e-6); y = d.y.values; acc = ((p > .5) == y).mean()
    return dict(acc=acc, nll=-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean(), over=np.maximum(p, 1 - p).mean() - acc)
vids = R.video.unique(); dv = [v for v in vids if v.startswith("DRONE")]; bv = [v for v in vids if v.startswith("BIRD")]
say(f"\n=== (3) segment fusion, complete 25-frame segments ({R.seg.nunique()} segments, {len(dv)} drone + {len(bv)} bird videos), "
    "video-level bootstrap 95% intervals (videos resampled within class) ===")
boot = {r: {"acc": [], "nll": [], "over": []} for r in R.rule.unique()}; diff = []
for _ in range(1000):
    pick = list(rng.choice(dv, len(dv))) + list(rng.choice(bv, len(bv)))
    Bt = pd.concat([R[R.video == v] for v in pick])
    ms = {r: met(Bt[Bt.rule == r]) for r in boot}
    for r in boot:
        for k in ("acc", "nll", "over"): boot[r][k].append(ms[r][k])
    diff.append(ms["independent over time"]["over"] - ms["design effect"]["over"])
for r in boot:
    m0 = met(R[R.rule == r]); ci = {k: np.quantile(boot[r][k], [.025, .975]) for k in boot[r]}
    say(f"  {r}: acc {m0['acc']:.3f} [{ci['acc'][0]:.3f}, {ci['acc'][1]:.3f}] | NLL {m0['nll']:.3f} [{ci['nll'][0]:.3f}, {ci['nll'][1]:.3f}] | "
        f"overconfidence {m0['over']:+.3f} [{ci['over'][0]:+.3f}, {ci['over'][1]:+.3f}]")
q = np.quantile(diff, [.025, .975])
say(f"  overconfidence reduction, independent minus design effect: {np.mean(diff):.3f} [{q[0]:.3f}, {q[1]:.3f}]")

# ------------------------------------------------------------ (4) Dawid-Skene audit, video bootstrap
def ds(Vm):
    w = (Vm.mean(1) > .5) * .8 + .1
    for _ in range(2000):
        pi = w.mean(); a = ((w[:, None] * Vm).sum(0) + .5) / (w.sum() + 1); b = (((1 - w)[:, None] * Vm).sum(0) + .5) / ((1 - w).sum() + 1)
        L1 = np.log(pi) + (Vm * np.log(a) + (1 - Vm) * np.log(1 - a)).sum(1); L0 = np.log(1 - pi) + (Vm * np.log(b) + (1 - Vm) * np.log(1 - b)).sum(1)
        wn = 1 / (1 + np.exp(L0 - L1))
        if np.abs(wn - w).max() < 1e-9: break
        w = wn
    return w.mean(), a, 1 - b
for c in ("ir", "rgb", "ir_t", "rgb_t"): D[c + "_v"] = (D[c] >= .25).astype(float).where(D[c].notna())
E = D.dropna(subset=["ir_t_v", "rgb_t_v"]); cols = ["ir_v", "rgb_v", "ir_t_v", "rgb_t_v"]
def audit_err(d):
    pi, se, sp = ds(d[cols].values)
    tse = [d.loc[d.truth == 1, c].mean() for c in cols]; tsp = [1 - d.loc[d.truth == 0, c].mean() for c in cols]
    return dict(prev=pi - d.truth.mean(), ir_se=se[0] - tse[0], vis_se=se[1] - tse[1], ir_sp=sp[0] - tsp[0], vis_sp=sp[1] - tsp[1])
e0 = audit_err(E); bs = {k: [] for k in e0}
ev = E.video.unique(); edv = [v for v in ev if v.startswith("DRONE")]; ebv = [v for v in ev if v.startswith("BIRD")]
for _ in range(500):
    pick = list(rng.choice(edv, len(edv))) + list(rng.choice(ebv, len(ebv)))
    r_ = audit_err(pd.concat([E[E.video == v] for v in pick]))
    for k in bs: bs[k].append(r_[k])
say("\n=== (4) AVID four-view Dawid-Skene audit error (estimate minus truth), video-level bootstrap 95% intervals ===")
for k, nm in (("prev", "prevalence"), ("ir_se", "IR sensitivity"), ("vis_se", "visible sensitivity"), ("ir_sp", "IR specificity"), ("vis_sp", "visible specificity")):
    q = np.quantile(bs[k], [.025, .975]); say(f"  {nm}: {e0[k] * 100:+.0f} points [{q[0] * 100:+.0f}, {q[1] * 100:+.0f}]")
# faster profile intervals: 7-point grid, then bisection (same precision at the endpoints)
def profile_ci_fast(S, which, est, nll0):
    cut = chi2.ppf(.95, 1) / 2; gap = lambda gv: _fit(S, **{which: gv})[0] - nll0 - cut
    grid = np.r_[0.0, np.geomspace(max(est, .05) / 4, max(est, .3) * 2.5, 6)]
    vals = np.array([gap(gv) for gv in grid]); inside = np.where(vals <= 0)[0]
    if len(inside) == 0: return (np.nan, np.nan), False
    def refine(a, b, fa):
        for _ in range(6):
            m = (a + b) / 2; fm = gap(m); a, b, fa = (m, b, fm) if (fm > 0) == (fa > 0) else (a, m, fa)
        return (a + b) / 2
    i0, i1 = inside[0], inside[-1]
    lo = 0.0 if i0 == 0 else refine(grid[i0 - 1], grid[i0], vals[i0 - 1])
    hi = grid[i1] if i1 == len(grid) - 1 else refine(grid[i1], grid[i1 + 1], vals[i1])
    return (lo, hi), i1 == len(grid) - 1
profile_ci = profile_ci_fast

# ------------------------------------------------------------ (1) fold-adjusted nested model
say("=== (1) 37 sessions, nested model with cross-fitting fold as a fixed effect ===")
for sen in A.SENS:
    d = V[V.sensor == sen]; t = d.groupby("track").size(); d = d[d.track.isin(t[t >= 3].index)]
    X = np.column_stack([np.ones(len(d)), np.log(d["size"].values), d.fold.values.astype(float)])
    clip = pd.factorize(d.track)[0]; sess = d.groupby("track").session.first(); scene = pd.factorize(d.track.map(sess))[0]
    n_clip, n_scene = clip.max() + 1, scene.max() + 1; cs = np.zeros(n_clip, int); cs[clip] = scene
    S = (X, d.err.values, clip, n_clip, n_scene, cs, d.session.nunique(), d.track.nunique())
    nll, s, tau = _fit(S); (slo, shi), _ = profile_ci(S, "sig", s, nll); (tlo, thi), _ = profile_ci(S, "tau", tau, nll)
    fold_rate = d.groupby("fold").err.mean().round(3).to_dict()
    say(f"{sen}: sigma_session {s:.2f} [{slo:.2f}, {shi:.2f}] | tau_sequence {tau:.2f} [{tlo:.2f}, {thi:.2f}] | "
        f"sequence share {tau**2 / (tau**2 + s**2):.2f} | miss rate by fold {fold_rate}")


say("[END] dce_checks finished")
