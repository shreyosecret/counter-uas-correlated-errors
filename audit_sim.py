"""Label-free audit under measured IR-visible dependence.
Views: IR detector, visible detector, motion cue, zero-shot VLM (one vote per track).
IR and visible errors are dependent within each true class with odds ratio OR;
motion and VLM are conditionally independent. Compare:
  DS   : Dawid-Skene (all four views conditionally independent)
  PAIR : dependence-aware LCM, IR-visible modeled as one 4-level joint view
  LAB  : direct estimate from a small labeled audit set (n = 30 tracks)
Error metric: mean absolute error (percentage points) of IR and visible
sensitivity and specificity, plus prevalence error."""
import numpy as np, sys
rng = np.random.default_rng(2026)
N, PREV, NLAB, REPS = 300, 0.6, 30, 300
# (sensitivity, specificity); IR, visible calibrated loosely to YOLO11n recall; motion, VLM to Halmstad
SE = dict(ir=.85, vis=.82, mot=.75, vlm=.60)
SP = dict(ir=.90, vis=.88, mot=.70, vlm=.95)

def p11(pa, pb, OR):
    """P(A=1,B=1) for marginals pa,pb and odds ratio OR (Plackett)."""
    if abs(OR - 1) < 1e-9: return pa * pb
    s = 1 + (pa + pb) * (OR - 1)
    return (s - np.sqrt(s * s - 4 * OR * (OR - 1) * pa * pb)) / (2 * (OR - 1))

def simulate(OR):
    y = rng.random(N) < PREV
    V = np.zeros((N, 4), int)
    for c in (0, 1):
        idx = np.where(y == c)[0]; n = len(idx)
        # error indicators for IR and vis, dependent with odds ratio OR
        eI = 1 - SE['ir'] if c else 1 - SP['ir']; eV = 1 - SE['vis'] if c else 1 - SP['vis']
        q11 = p11(eI, eV, OR); probs = [q11, eI - q11, eV - q11, 1 - eI - eV + q11]
        cell = rng.choice(4, n, p=probs)
        errI = np.isin(cell, [0, 1]); errV = np.isin(cell, [0, 2])
        V[idx, 0] = np.where(errI, 1 - c, c); V[idx, 1] = np.where(errV, 1 - c, c)
        for j, k in ((2, 'mot'), (3, 'vlm')):
            p1 = SE[k] if c else 1 - SP[k]
            V[idx, j] = rng.random(n) < p1
    return y.astype(int), V

def em(V, pair, iters=500):
    """Latent class EM. pair=True: views 0,1 enter as one 4-level categorical."""
    n = len(V)
    w = (V.mean(1) > 0.5).astype(float) * 0.8 + 0.1          # init from majority vote
    pc = V[:, 0] * 2 + V[:, 1]                                 # joint IR-vis code 0..3
    for _ in range(iters):
        pi = w.mean()
        def rate(col): return ((w * col).sum() + .5) / (w.sum() + 1), (((1 - w) * col).sum() + .5) / ((1 - w).sum() + 1)
        L1 = np.full(n, np.log(pi)); L0 = np.full(n, np.log(1 - pi))
        if pair:
            J1 = np.array([((w * (pc == k)).sum() + .25) / (w.sum() + 1) for k in range(4)])
            J0 = np.array([(((1 - w) * (pc == k)).sum() + .25) / ((1 - w).sum() + 1) for k in range(4)])
            L1 += np.log(J1[pc]); L0 += np.log(J0[pc]); rest = (2, 3)
        else:
            rest = (0, 1, 2, 3)
        r = {}
        for j in rest:
            a, b = rate(V[:, j]); r[j] = (a, b)
            L1 += np.where(V[:, j], np.log(a), np.log(1 - a)); L0 += np.where(V[:, j], np.log(b), np.log(1 - b))
        w_new = 1 / (1 + np.exp(L0 - L1))
        if np.abs(w_new - w).max() < 1e-7: w = w_new; break
        w = w_new
    pi = w.mean()
    if pair:  # marginal IR / vis rates from the joint table
        se_ir, fp_ir = J1[2] + J1[3], J0[2] + J0[3]; se_v, fp_v = J1[1] + J1[3], J0[1] + J0[3]
    else:
        se_ir, fp_ir = r[0]; se_v, fp_v = r[1]
    return pi, se_ir, 1 - fp_ir, se_v, 1 - fp_v

def labeled(y, V):
    i = rng.choice(len(y), NLAB, replace=False); yy, VV = y[i], V[i]
    f = lambda col, c: ((VV[yy == c, col] == c).sum() + .5) / ((yy == c).sum() + 1)
    return yy.mean(), f(0, 1), f(0, 0), f(1, 1), f(1, 0)

truth = np.array([PREV, SE['ir'], SP['ir'], SE['vis'], SP['vis']])
print("OR | method | MAE IR/vis Se,Sp (pts) [90% range] | prevalence err (pts) | IR Se bias | vis Sp bias")
rows = []
for OR in (1.0, 2.5, 8.0):
    res = {m: [] for m in ("DS", "PAIR", "LAB")}
    for _ in range(REPS):
        y, V = simulate(OR)
        for m, f in (("DS", lambda: em(V, False)), ("PAIR", lambda: em(V, True)), ("LAB", lambda: labeled(y, V))):
            res[m].append(np.array(f()) - truth)
    for m, R in res.items():
        R = np.array(R) * 100; mae = np.abs(R[:, 1:]).mean(1)
        rows.append((OR, m, mae.mean(), *np.quantile(mae, [.05, .95]), np.abs(R[:, 0]).mean(), R[:, 1].mean(), R[:, 4].mean()))
        print(f"{OR:3} | {m:4} | {mae.mean():5.1f} [{np.quantile(mae,.05):4.1f}, {np.quantile(mae,.95):4.1f}] | {np.abs(R[:,0]).mean():4.1f} | {R[:,1].mean():+5.1f} | {R[:,4].mean():+5.1f}")
np.save("audit_rows.npy", np.array([[r[0]] + list(r[2:]) for r in rows]))
