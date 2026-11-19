"""
Bridge experiment: run the label-free latent class audit on REAL votes and
check its sensor-accuracy estimates against ground truth.

Items = clips (one target class each). Views, one vote per clip:
  HOG      HOG + logistic, video-grouped CV, majority over the clip's crops
  CLIP     zero-shot CLIP, majority over crops        (clip_<s>.csv)
  PROBE    linear probe on CLIP features, majority    (probe_<s>.csv)
  QWEN     Qwen2.5-VL-3B yes/no prompt, majority      (qwen2_<s>.csv)
  MOTION   track-level motion classifier, CV
Models (no labels used):
  majority vote
  DS       Dawid-Skene / Hui-Walter latent class model (conditional independence)
  RE-LCM   Qu-Tan-Kutner random-effects latent class model: appearance views
           share a hard-clip effect u; MOTION is left independent
Outputs: true vs estimated sensitivity and specificity per view, label
accuracy, pairwise dependence residuals under DS, and the DS vs RE-LCM
likelihood-ratio statistic.
Usage:  python audit_real.py <data root> V|IR      (CSV files in the working dir)
"""
import sys, warnings
import numpy as np, pandas as pd
from scipy.special import expit
warnings.filterwarnings("ignore")
NAMES = ["HOG", "CLIP", "PROBE", "QWEN", "MOTION"]
APPEARANCE = (0, 1, 2, 3)


# ---------------------------------------------------------------- models
def ds_fit(V, iters=500):
    n, k = V.shape; obs = V >= 0
    s = (V == 1).sum(1); m = obs.sum(1)
    post = np.clip((s + 0.5) / (m + 1), 0.05, 0.95)
    for _ in range(iters):
        pi = post.mean()
        se = np.array([((post * (V[:, v] == 1))[obs[:, v]].sum() + 1) / (post[obs[:, v]].sum() + 2) for v in range(k)])
        sp = np.array([(((1 - post) * (V[:, v] == 0))[obs[:, v]].sum() + 1) / ((1 - post)[obs[:, v]].sum() + 2) for v in range(k)])
        l1 = np.log(pi) + np.where(obs, np.where(V == 1, np.log(se), np.log(1 - se)), 0).sum(1)
        l0 = np.log(1 - pi) + np.where(obs, np.where(V == 0, np.log(sp), np.log(1 - sp)), 0).sum(1)
        mx = np.maximum(l1, l0); ll = float((mx + np.log(np.exp(l1 - mx) + np.exp(l0 - mx))).sum())
        post = expit(l1 - l0)
    return post, se, sp, pi, ll


def re_fit(V, u_views=APPEARANCE, Q=15, iters=400, newton=3, lam_prior_sd=1.0):
    """logit P(view v correct | class c, u) = a[v,c] + lam[v]*u, u ~ N(0,1) per clip.
    Loadings get a N(0, lam_prior_sd^2) penalty: with ~150 clips and four correlated
    views they are weakly identified, and unpenalised EM drives them to extremes."""
    ridge = np.array([1e-2, 1e-2, 1.0 / lam_prior_sd ** 2])
    n, k = V.shape; obs = V >= 0
    x, w = np.polynomial.hermite.hermgauss(Q); uq = np.sqrt(2) * x; lw = np.log(w / np.sqrt(np.pi))
    th = np.array([[1.0, 1.0, (-0.5 if v in u_views else 0.0)] for v in range(k)])   # a0, a1, lam
    pi = 0.5; ll = None
    for it in range(iters):
        # E-step over (class, node)
        L = np.zeros((n, 2, Q)) + lw[None, None, :]
        L[:, 0] += np.log(1 - pi); L[:, 1] += np.log(pi)
        for v in range(k):
            for c in (0, 1):
                pc = np.clip(expit(th[v, c] + th[v, 2] * uq), 1e-6, 1 - 1e-6)[None, :]
                corr = (V[:, v] == c)[:, None]
                L[:, c] += np.where(obs[:, v][:, None], np.where(corr, np.log(pc), np.log(1 - pc)), 0)
        mx = L.max(axis=(1, 2), keepdims=True)
        lse = mx.squeeze() + np.log(np.exp(L - mx).sum((1, 2)))
        ll = float(lse.sum())
        post = np.exp(L - lse[:, None, None])                       # (n,2,Q)
        # M-step
        pi = float(np.clip(post[:, 1].sum() / n, 0.01, 0.99))
        for v in range(k):
            o = obs[:, v]; W = post[o]                                # (m,2,Q)
            m = o.sum()
            C = np.broadcast_to(np.array([0, 1])[None, :, None], (m, 2, Q))
            U = np.broadcast_to(uq[None, None, :], (m, 2, Q))
            X = np.stack([(C == 0), (C == 1), U if v in u_views else np.zeros_like(U)], -1).reshape(-1, 3).astype(float)
            t = (V[o, v][:, None, None] == C).reshape(-1).astype(float); ww = W.reshape(-1)
            b = th[v].copy()
            for _ in range(newton):
                p = expit(X @ b)
                g = X.T @ (ww * (t - p)) - ridge * b
                H = (X * (ww * p * (1 - p))[:, None]).T @ X + np.diag(ridge)
                b = b + np.linalg.solve(H, g)
            if v not in u_views:
                b[2] = 0.0
            th[v] = b
    wq = np.exp(lw)
    se = np.array([(wq * expit(th[v, 1] + th[v, 2] * uq)).sum() for v in range(k)])
    sp = np.array([(wq * expit(th[v, 0] + th[v, 2] * uq)).sum() for v in range(k)])
    return post[:, 1].sum(1), se, sp, pi, ll, th


def ds_residuals(V, se, sp, pi):
    """Observed vs model-implied pairwise agreement under conditional independence."""
    out = {}
    k = V.shape[1]
    for a in range(k):
        for b in range(a + 1, k):
            o = (V[:, a] >= 0) & (V[:, b] >= 0)
            e = pi * (se[a] * se[b] + (1 - se[a]) * (1 - se[b])) + (1 - pi) * (sp[a] * sp[b] + (1 - sp[a]) * (1 - sp[b]))
            obs_ag = (V[o, a] == V[o, b]).mean(); m = o.sum()
            out[f"{NAMES[a]}-{NAMES[b]}"] = (obs_ag - e) / np.sqrt(e * (1 - e) / m)
    return out


# ---------------------------------------------------------------- build votes
def build_votes(root, s):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from halmstad_lib import extract_crops
    tracks, crops = extract_crops(root, s, hog_feats=True)
    T = pd.DataFrame(tracks).dropna(); C = pd.DataFrame(crops)
    feats = ["speed", "straight", "area_flicker", "aspect_flicker", "size_var"]
    mot = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced"))
    T["MOTION"] = cross_val_predict(mot, T[feats].values, T.y.values, cv=StratifiedKFold(5, shuffle=True, random_state=0))
    X = np.vstack(C.feat.values)
    hog = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, class_weight="balanced", max_iter=3000))
    C["HOG"] = cross_val_predict(hog, X, C.y.values, cv=GroupKFold(5), groups=C.track.values)
    C["file"] = C.track + "__f" + C.frame.astype(str) + ".jpg"
    for name, f in [("CLIP", f"clip_{s}.csv"), ("PROBE", f"probe_{s}.csv"), ("QWEN", f"qwen2_{s}.csv")]:
        C = C.merge(pd.read_csv(f)[["file", "vote"]].rename(columns={"vote": name}), on="file", how="left")
    agg = C.groupby("track")[["HOG", "CLIP", "PROBE", "QWEN"]].mean()
    agg = (agg >= 0.5).astype(int).where(agg.notna(), -1)
    D = T.set_index("track")[["y", "MOTION"]].join(agg, how="inner")
    return D


def report(D, s):
    V = D[NAMES].values.astype(int); y = D.y.values
    tse = np.array([(V[y == 1, v] == 1).mean() for v in range(5)])
    tsp = np.array([(V[y == 0, v] == 0).mean() for v in range(5)])
    pd_, dse, dsp, dpi, dll = ds_fit(V)
    pr, rse, rsp, rpi, rll, th = re_fit(V)
    maj = ((V == 1).sum(1) * 2 > (V >= 0).sum(1)).astype(int)
    print(f"\n=== {s}: {len(D)} clips ({y.sum()} drone, {(y == 0).sum()} bird); true drone share {y.mean():.3f}")
    print(f"estimated drone share: DS {dpi:.3f}, RE-LCM {rpi:.3f}")
    print("view    | sensitivity true / DS / RE | specificity true / DS / RE")
    for v in range(5):
        print(f"{NAMES[v]:7s} | {tse[v]:.3f} / {dse[v]:.3f} / {rse[v]:.3f}      | {tsp[v]:.3f} / {dsp[v]:.3f} / {rsp[v]:.3f}")
    mae = lambda a, b: 100 * np.mean(np.abs(np.r_[a] - np.r_[b]))
    print(f"audit error (mean abs, pts over 10 numbers): DS {mae(np.r_[dse, dsp], np.r_[tse, tsp]):.1f}, "
          f"RE-LCM {mae(np.r_[rse, rsp], np.r_[tse, tsp]):.1f}")
    best = max(range(5), key=lambda v: (V[:, v] == y).mean())
    print(f"clip label accuracy: majority {(maj == y).mean():.3f}, DS {((pd_ > .5) == y).mean():.3f}, "
          f"RE-LCM {((pr > .5) == y).mean():.3f}, best single view ({NAMES[best]}) {(V[:, best] == y).mean():.3f}")
    print(f"likelihood ratio RE-LCM vs DS: {2 * (rll - dll):.1f}  (4 extra loading parameters)")
    print("loadings (appearance views):", {NAMES[v]: round(float(th[v, 2]), 2) for v in APPEARANCE})
    z = ds_residuals(V, dse, dsp, dpi)
    print("DS pairwise agreement residuals (z > 2 = unmodelled dependence):",
          {k: round(float(x), 1) for k, x in sorted(z.items(), key=lambda kv: -kv[1])[:4]})
    tc = {}
    for a in range(5):
        for b in range(a + 1, 5):
            r = []
            for c in (0, 1):
                ea, eb = (V[y == c, a] != c).astype(float), (V[y == c, b] != c).astype(float)
                if ea.std() > 0 and eb.std() > 0:
                    r.append(np.corrcoef(ea, eb)[0, 1])
            tc[f"{NAMES[a]}-{NAMES[b]}"] = np.mean(r) if r else np.nan
    print("TRUE within-class error correlation, top pairs:",
          {k: round(float(x), 2) for k, x in sorted(tc.items(), key=lambda kv: -np.nan_to_num(kv[1]))[:4]})


if __name__ == "__main__":
    root, s = sys.argv[1], sys.argv[2]
    report(build_votes(root, s), s)
