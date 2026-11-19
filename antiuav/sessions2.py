"""
Fixing the few-clusters problem for session-level estimates (Anti-UAV RGB-T).
  PROFILE : profile-likelihood 95% CIs for sigma (session) and tau (sequence) on the existing 15-session data.
  DATA    : frames every 50 from ALL sequences, both sensors, YOLO labels (local disk; small metadata to Drive).
  TRAIN   : 2-fold session cross-fitting: sessions split in two halves; a YOLO11n per half (checkpoints on Drive).
  SCORE   : each half scored by the detector that never saw it -> out-of-session errors for every sequence.
  ANALYZE : nested session/sequence model over all sessions with profile CIs; cross-sensor dependence.
Run on a T4:
    !pip -q install ultralytics statsmodels huggingface_hub
    from google.colab import drive; drive.mount('/content/drive')
    %run /content/drive/MyDrive/counter_uas_attack/sessions2.py
"""
import os, sys, json, math, glob, random, shutil, traceback
import numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.stats import chi2
WORK = "/content/drive/MyDrive/counter_uas_attack"; LOC = "/content/sess"; os.makedirs(LOC, exist_ok=True)
sys.path.insert(0, WORK); import attack as A
OUT = f"{WORK}/sessions_results.txt"; EPOCHS = 15
lines = open(OUT).read().split("\n") if os.path.exists(OUT) else []
def say(x):
    print(x, flush=True); lines.append(str(x)); open(OUT, "w").write("\n".join(lines))
def stage(tag, fn):
    if f"[{tag}] DONE" in lines: print(tag, "done"); return
    try:
        if fn() is not False: say(f"[{tag}] DONE")
    except Exception as e: say(f"[{tag}] FAILED (rerun retries): {e!r}"); traceback.print_exc(); raise

# ------------------------------------------------------------------ profile likelihood for variance components
def _setup(d):
    t = d.groupby("track").size(); d = d[d.track.isin(t[t >= 3].index)]
    X = A.design(d); y = d.err.values; clip = pd.factorize(d.track)[0]
    sess = d.groupby("track").session.first(); scene = pd.factorize(d.track.map(sess))[0]
    n_clip, n_scene = clip.max() + 1, scene.max() + 1; cs = np.zeros(n_clip, int); cs[clip] = scene
    return X, y, clip, n_clip, n_scene, cs, d.session.nunique(), d.track.nunique()

def _fit(S, sig=None, tau=None):
    X, y, clip, n_clip, n_scene, cs = S[:6]; k = X.shape[1]
    def f(p):
        i = k; s = sig if sig is not None else np.exp(p[i]); i += sig is None
        t = tau if tau is not None else np.exp(p[i])
        return A.negll2(p[:k], s, t, X, y, clip, n_clip, n_scene, cs)
    best = None
    for s0, t0 in ((.5, 1.), (1.5, .5), (.2, .2)):
        p0 = np.r_[np.zeros(k), [np.log(s0)] if sig is None else [], [np.log(t0)] if tau is None else []]
        r = minimize(f, p0, method="BFGS")
        if best is None or r.fun < best.fun: best = r
    p = best.x; i = k
    s = sig if sig is not None else float(np.exp(p[i])); i += sig is None
    t = tau if tau is not None else float(np.exp(p[i]))
    return best.fun, s, t

def profile_ci(S, which, est, nll0):
    """95% profile-likelihood CI; grid search, then bisection to refine each endpoint."""
    cut = chi2.ppf(.95, 1) / 2
    gap = lambda g: _fit(S, **{which: g})[0] - nll0 - cut
    grid = np.r_[0.0, np.geomspace(max(est, .05) / 8, max(est, .3) * 4, 13)]
    vals = np.array([gap(g) for g in grid]); inside = np.where(vals <= 0)[0]
    if len(inside) == 0: return (np.nan, np.nan), False
    def refine(a, b, fa):                       # root of gap between a and b, fa = gap(a)
        for _ in range(7):
            m = (a + b) / 2; fm = gap(m)
            a, b, fa = (m, b, fm) if (fm > 0) == (fa > 0) else (a, m, fa)
        return (a + b) / 2
    i0, i1 = inside[0], inside[-1]
    lo = 0.0 if i0 == 0 else refine(grid[i0 - 1], grid[i0], vals[i0 - 1])
    hi = grid[i1] if i1 == len(grid) - 1 else refine(grid[i1], grid[i1 + 1], vals[i1])
    return (lo, hi), i1 == len(grid) - 1

def nested_report(d, label):
    S = _setup(d); nll, s, t = _fit(S)
    (slo, shi), sopen = profile_ci(S, "sig", s, nll); (tlo, thi), topen = profile_ci(S, "tau", t, nll)
    share = t ** 2 / (t ** 2 + s ** 2 + 1e-12)
    say(f"{label}: sigma_session {s:.2f} [{slo:.2f}, {shi:.2f}{'+' if sopen else ''}] | tau_sequence {t:.2f} [{tlo:.2f}, {thi:.2f}{'+' if topen else ''}] "
        f"| sequence share {share:.2f} | {S[7]} seqs, {S[6]} sessions  (95% profile CIs; '+' = upper end beyond grid)")

def profile_existing():
    VA = pd.read_csv(f"{WORK}/attack_votes_spread.csv", dtype={"track": str, "session": str})
    TB, FA = VA[VA.split == "test"], VA[VA.split == "far"]
    say("=== PROFILE: existing 15-session data (test set B + far range), YOLO11n ===")
    for sen in A.SENS: nested_report(pd.concat([TB, FA])[lambda d: d.sensor == sen], f"[PROFILE] {sen}")
stage("PROFILE", profile_existing)

# ------------------------------------------------------------------ cross-fitting over all sessions
from huggingface_hub import hf_hub_download
import cv2
from huggingface_hub import list_repo_files
AJ = f"{WORK}/annotations.json" if os.path.exists(f"{WORK}/annotations.json") else f"{LOC}/annotations.json"
if os.path.exists(AJ): ANN = json.load(open(AJ))
else:                                                     # cache was deleted: rebuild it on local disk
    seqs = sorted({"/".join(f.split("/")[:-1]) for f in list_repo_files(A.REPO, repo_type="dataset") if f.endswith("infrared.json")})
    ANN = {}
    for k, sq in enumerate(seqs):
        try: ANN[sq] = {m: json.load(open(hf_hub_download(A.REPO, f"{sq}/{m}.json", repo_type="dataset"))) for m in A.SENS}
        except Exception as e: print("skip", sq, e)
        if k % 50 == 0: print("annotations", k, "/", len(seqs), flush=True)
    json.dump(ANN, open(AJ, "w"))
FULL = sorted(ANN)
META = f"{WORK}/xfit_meta.csv"
def ok(a, m, f): return f < len(a[m]["exist"]) and a[m]["exist"][f] and len(a[m]["gt_rect"][f]) == 4 and a[m]["gt_rect"][f][2] * a[m]["gt_rect"][f][3] > 0
def frames(s, step=50):
    a = ANN[s]; n = min(len(a["infrared"]["exist"]), len(a["visible"]["exist"]))
    return [f for f in range(0, n, step) if ok(a, "infrared", f) and ok(a, "visible", f)]
sessions = sorted({A.session(s) for s in FULL}); rnd = random.Random(7); rnd.shuffle(sessions)
FOLD = {ss: i % 2 for i, ss in enumerate(sessions)}

def data():
    os.makedirs(f"{LOC}/images", exist_ok=True); os.makedirs(f"{LOC}/labels", exist_ok=True); rows = []
    for k, s in enumerate(FULL):
        want = set(frames(s))
        if len(want) < 3: continue
        for m in A.SENS:
            vid = hf_hub_download(A.REPO, f"{s}/{m}.mp4", repo_type="dataset"); cap = cv2.VideoCapture(vid); f = 0; todo = set(want)
            while todo:
                good, img = cap.read()
                if not good: break
                if f in todo:
                    H, W = img.shape[:2]; x, y, w, h = ANN[s][m]["gt_rect"][f]; p = f"{LOC}/images/{s.replace('/', '_')}__{m}__{f}.jpg"
                    cv2.imwrite(p, img); open(p.replace("/images/", "/labels/").replace(".jpg", ".txt"), "w").write(
                        f"0 {(x + w / 2) / W:.6f} {(y + h / 2) / H:.6f} {w / W:.6f} {h / H:.6f}\n")
                    rows.append(dict(img=p, track=s, session=A.session(s), fold=FOLD[A.session(s)], frame=f, sensor=m, x=x, y=y, w=w, h=h)); todo.discard(f)
                f += 1
            cap.release()
            try: os.remove(os.path.realpath(vid))
            except Exception: pass
        if k % 20 == 0: print("extract", k, "/", len(FULL), len(rows), flush=True)
    M = pd.DataFrame(rows); M.to_csv(META, index=False)
    say(f"[DATA] {M.track.nunique()} sequences, {M.session.nunique()} sessions ({(M.groupby('session').fold.first() == 0).sum()} / "
        f"{(M.groupby('session').fold.first() == 1).sum()} per fold), {len(M)} images")
if not os.path.exists(f"{LOC}/images") or not os.listdir(f"{LOC}/images"):
    lines[:] = [l for l in lines if not l.startswith("[DATA]")]; stage("DATA", data)

def train():
    from ultralytics import YOLO
    M = pd.read_csv(META, dtype={"track": str, "session": str})
    for fd in (0, 1):
        rd = f"{WORK}/runs/xfit_{fd}"; rc = f"{rd}/results.csv"
        n = lambda: len(pd.read_csv(rc)) if os.path.exists(rc) else 0
        if n() >= EPOCHS: continue
        T = M[M.fold == fd]; tr_tracks = sorted(T.track.unique()); rnd2 = random.Random(fd); rnd2.shuffle(tr_tracks)
        val = set(tr_tracks[:max(4, len(tr_tracks) // 10)])
        tr_imgs = T[~T.track.isin(val) & (T.frame % 100 == 0)].img; va_imgs = T[T.track.isin(val)].img
        open(f"{LOC}/train_{fd}.txt", "w").write("\n".join(tr_imgs)); open(f"{LOC}/val_{fd}.txt", "w").write("\n".join(va_imgs))
        open(f"{LOC}/xfit_{fd}.yaml", "w").write(f"path: {LOC}\ntrain: {LOC}/train_{fd}.txt\nval: {LOC}/val_{fd}.txt\nnames:\n  0: drone\n")
        os.chdir(LOC)
        if os.path.exists(f"{rd}/weights/last.pt"):
            try: YOLO(f"{rd}/weights/last.pt").train(resume=True)
            except Exception:
                if n() < EPOCHS: raise
        else:
            YOLO("yolo11n.pt").train(data=f"{LOC}/xfit_{fd}.yaml", epochs=EPOCHS, imgsz=640, batch=32, project=f"{WORK}/runs",
                                     name=f"xfit_{fd}", exist_ok=True, verbose=False, plots=False, workers=4)
        say(f"[TRAIN] fold {fd}: {len(tr_imgs)} training images, val mAP50 {pd.read_csv(rc).filter(like='mAP50(B)').iloc[-1, 0]:.3f}")
stage("TRAIN", train)

VOTES = f"{WORK}/xfit_votes.csv"
def score():
    from ultralytics import YOLO
    if os.path.exists(VOTES): return
    M = pd.read_csv(META, dtype={"track": str, "session": str}); parts = []
    for fd in (0, 1):
        model = YOLO(f"{WORK}/runs/xfit_{fd}/weights/best.pt"); D = M[M.fold != fd].copy(); hits = []
        for i in range(0, len(D), 32):
            ch = D.iloc[i:i + 32]
            for (_, r), res in zip(ch.iterrows(), model.predict(list(ch.img), imgsz=640, conf=0.25, verbose=False)):
                b = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else np.zeros((0, 4))
                cx, cy = r.x + r.w / 2, r.y + r.h / 2; tol = max(r.w, r.h, 4)
                hits.append(int(any(abs((b0 + b2) / 2 - cx) <= tol and abs((b1 + b3) / 2 - cy) <= tol for b0, b1, b2, b3 in b)))
        D["err"] = 1 - np.array(hits); D["size"] = np.sqrt(D.w * D.h); parts.append(D)
    V = pd.concat(parts); V.drop(columns=["img"]).to_csv(VOTES, index=False)
    say(f"[SCORE] {len(V)} out-of-session predictions; recall IR {1 - V[V.sensor == 'infrared'].err.mean():.3f}, visible {1 - V[V.sensor == 'visible'].err.mean():.3f}")
stage("SCORE", score)

def analyze():
    from statsmodels.stats.contingency_tables import StratifiedTable
    V = pd.read_csv(VOTES, dtype={"track": str, "session": str})
    say("=== ANALYZE: all sessions, out-of-session (cross-fitted) YOLO11n ===")
    for sen in A.SENS:
        say("[ANALYZE] " + A.tau_line(V[V.sensor == sen], f"{sen} single-level")[0])
        nested_report(V[V.sensor == sen], f"[ANALYZE] {sen} nested")
    say("[ANALYZE] " + A.cross_line(V, "cross-sensor, all sessions", B=1000)[0])
stage("ANALYZE", analyze)
say("[END] sessions.py finished")
