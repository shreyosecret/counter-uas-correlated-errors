"""
Closing the three reviewer gaps, on Anti-UAV RGB-T (public copy on Hugging Face).

1. A TRAINED DRONE DETECTOR. YOLO11n is fine-tuned on train-split sequences
   (both sensors) and evaluated on held-out test-split sequences and on
   far-range sequences excluded from training. Error = annotated drone missed.
   Training sequences are chosen, where possible, from recording sessions that
   appear in neither evaluation set, and any remaining overlap is reported.
2. REAL SCENE LABELS. Sequence names encode the recording session (date_time):
       logit P(miss) = X b + sigma * v_session + tau * u_sequence
3. MORE FAR-RANGE DATA. Far-range sequences have >= 2 frames (sampled every
   25 frames) with infrared box diagonal <= 30 px. Results are reported two ways:
   all frames of those sequences, and only the frames that are actually far.
   Cross-sensor dependence: Mantel-Haenszel OR within sequences, plus per-sequence
   correlation with sequence-level and session-level bootstrap CIs.

HOW TO RUN (Google Colab, T4 GPU):
  1. Put this file in Google Drive at MyDrive/counter_uas_attack/attack.py
  2. In one cell:
        from google.colab import drive; drive.mount('/content/drive')
        %run /content/drive/MyDrive/counter_uas_attack/attack.py 40 40 25
  3. If the runtime disconnects: reconnect on T4 and run the same cell again.
     Frames, the split, training epochs and detector evaluation all resume.
  4. Results: MyDrive/counter_uas_attack/attack_results.txt (and .json)
Arguments: training sequences, test sequences, epochs. Keep them the same on reruns.
"""
import os, sys, json, math, glob, shutil, warnings, subprocess
import numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.special import log_expit
from scipy.stats import chi2
warnings.filterwarnings("ignore")

REPO = "CornBac0n/Anti-UAV-RGBT"
SENS = ("infrared", "visible")
FAR_PX = 30


def session(s):
    return s.split("/")[-1].rsplit("_", 2)[0]


# =====================================================================
# statistics: exact-likelihood random-intercept logistic models
# =====================================================================
XQ, WQ = np.polynomial.hermite.hermgauss(30)
UQ, LWQ = np.sqrt(2) * XQ, np.log(WQ / np.sqrt(np.pi))


def design(d):
    return np.column_stack([np.ones(len(d)), np.log(d["size"].values)])


def negll(b, tau, X, y, g, ng):
    eta = (X @ b)[:, None] + tau * UQ[None, :]
    lp = np.where(y[:, None] == 1, log_expit(eta), log_expit(-eta))
    S = np.zeros((ng, len(UQ))); np.add.at(S, g, lp)
    S = S + LWQ[None, :]
    m = S.max(1, keepdims=True)
    return -float((m.squeeze(1) + np.log(np.exp(S - m).sum(1))).sum())


def fit_b(tau, X, y, g, ng, b0):
    r = minimize(negll, b0, args=(tau, X, y, g, ng), method="BFGS")
    return r.x, r.fun


def fit1(X, y, g):
    ng = g.max() + 1; best = None
    for t0 in (0.3, 1.0, 2.0):
        r = minimize(lambda p: negll(p[:-1], np.exp(p[-1]), X, y, g, ng),
                     np.r_[np.zeros(X.shape[1]), np.log(t0)], method="BFGS")
        if best is None or r.fun < best.fun:
            best = r
    tau, b = float(np.exp(best.x[-1])), best.x[:-1]
    _, nll0 = fit_b(0.0, X, y, g, ng, b)
    return tau, b, best.fun, max(0.0, 2 * (nll0 - best.fun))


def profile_ci(X, y, g, tau, b, nll, level=0.95):
    ng = g.max() + 1; cut = chi2.ppf(level, 1) / 2
    gap = lambda t: fit_b(t, X, y, g, ng, b)[1] - nll - cut
    lo = 0.0
    if gap(1e-6) > 0:
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


Q2 = 15
_x2, _w2 = np.polynomial.hermite.hermgauss(Q2)
UQ2, LWQ2 = np.sqrt(2) * _x2, np.log(_w2 / np.sqrt(np.pi))


def negll2(b, sig, tau, X, y, clip, n_clip, n_scene, clip_scene):
    eta = (X @ b)[:, None, None] + sig * UQ2[None, :, None] + tau * UQ2[None, None, :]
    lp = np.where(y[:, None, None] == 1, log_expit(eta), log_expit(-eta))
    S = np.zeros((n_clip, Q2, Q2)); np.add.at(S, clip, lp)
    S = S + LWQ2[None, None, :]
    m = S.max(2, keepdims=True); Lc = (m + np.log(np.exp(S - m).sum(2, keepdims=True))).squeeze(2)
    T = np.zeros((n_scene, Q2)); np.add.at(T, clip_scene, Lc)
    T = T + LWQ2[None, :]
    m2 = T.max(1, keepdims=True)
    return -float((m2.squeeze(1) + np.log(np.exp(T - m2).sum(1))).sum())


def fit2(X, y, clip, scene, fix_tau0=False, fix_sig0=False):
    n_clip, n_scene = clip.max() + 1, scene.max() + 1
    clip_scene = np.zeros(n_clip, int); clip_scene[clip] = scene
    k = X.shape[1]

    def f(p):
        s = 0.0 if fix_sig0 else np.exp(p[k]); t = 0.0 if fix_tau0 else np.exp(p[k + 1])
        return negll2(p[:k], s, t, X, y, clip, n_clip, n_scene, clip_scene)
    best = None
    for s0, t0 in [(0.5, 1.0), (1.0, 0.5), (0.3, 0.3)]:
        r = minimize(f, np.r_[np.zeros(k), np.log(s0), np.log(t0)], method="BFGS")
        if best is None or r.fun < best.fun:
            best = r
    p = best.x
    return dict(sig=0.0 if fix_sig0 else float(np.exp(p[k])),
                tau=0.0 if fix_tau0 else float(np.exp(p[k + 1])), nll=float(best.fun))


# =====================================================================
# report lines (each returns text and a dict for the JSON summary)
# =====================================================================
def keep_tracks(d, min_n):
    t = d.groupby("track").size()
    return d[d.track.isin(t[t >= min_n].index)]


def tau_line(d, label, min_n=3):
    d = keep_tracks(d, min_n)
    if len(d) < 30 or d.err.nunique() < 2:
        return f"{label}: recall {1 - d.err.mean():.3f}, tau not estimable ({d.track.nunique()} seqs, {len(d)} frames)", \
               dict(label=label, recall=float(1 - d.err.mean()) if len(d) else None, tau=None)
    X = design(d); g = pd.factorize(d.track)[0]; y = d.err.values
    tau, b, nll, lr = fit1(X, y, g); lo, hi = profile_ci(X, y, g, tau, b, nll)
    p = 0.5 * chi2.sf(lr, 1)
    return (f"{label}: recall {1 - d.err.mean():.3f} | tau {tau:.2f} [{lo:.2f}, {hi:.2f}] p={p:.2g} "
            f"| {d.track.nunique()} seqs, {d.session.nunique()} sessions, {len(d)} frames"), \
        dict(label=label, recall=float(1 - d.err.mean()), tau=tau, ci=[lo, hi], p=p,
             seqs=int(d.track.nunique()), sessions=int(d.session.nunique()), frames=int(len(d)))


def session_line(d, label, min_n=3):
    d = keep_tracks(d, min_n)
    if d.session.nunique() < 3 or d.err.nunique() < 2:
        return f"{label}: too few sessions for the split ({d.session.nunique()})", dict(label=label, tau=None)
    X = design(d); y = d.err.values; clip = pd.factorize(d.track)[0]
    sess_of = d.groupby("track").session.first(); scene = pd.factorize(d.track.map(sess_of))[0]
    F = fit2(X, y, clip, scene); NT = fit2(X, y, clip, scene, fix_tau0=True); NS = fit2(X, y, clip, scene, fix_sig0=True)
    p_t = 0.5 * chi2.sf(max(0, 2 * (NT["nll"] - F["nll"])), 1)
    p_s = 0.5 * chi2.sf(max(0, 2 * (NS["nll"] - F["nll"])), 1)
    share = F["tau"] ** 2 / (F["tau"] ** 2 + F["sig"] ** 2 + 1e-12)
    return (f"{label}: tau_sequence {F['tau']:.2f} (p={p_t:.2g}) | sigma_session {F['sig']:.2f} (p={p_s:.2g}) "
            f"| sequence share of clustering variance {share:.2f} | {d.track.nunique()} seqs, {d.session.nunique()} sessions"), \
        dict(label=label, tau_seq=F["tau"], p_seq=p_t, sigma_session=F["sig"], p_session=p_s, seq_share=share)


def _boot_corr(r, groups, B, rng):
    """Correlation of per-sequence IR and visible miss rates, resampling whole groups."""
    keys = np.array(sorted(set(groups)))
    idx = {k: np.where(groups == k)[0] for k in keys}
    out = []
    for _ in range(B):
        pick = np.concatenate([idx[k] for k in rng.choice(keys, len(keys), replace=True)])
        k = r.iloc[pick]
        if k.eI.std() > 0 and k.eV.std() > 0:
            out.append(k.eI.corr(k.eV))
    return (np.quantile(out, .025), np.quantile(out, .975)) if len(out) > 50 else (np.nan, np.nan)


def cross_line(d, label, B=2000, seed=0):
    from statsmodels.stats.contingency_tables import StratifiedTable
    P = d.pivot_table(index=["track", "frame"], columns="sensor", values="err").dropna()
    if len(P) < 20:
        return f"{label}: too few paired frames ({len(P)})", dict(label=label)
    E = pd.DataFrame(dict(eI=P["infrared"].astype(int), eV=P["visible"].astype(int)))
    tabs = []
    for _, g in E.groupby(level=0):
        tb = np.array([[((g.eI == 1) & (g.eV == 1)).sum(), ((g.eI == 1) & (g.eV == 0)).sum()],
                       [((g.eI == 0) & (g.eV == 1)).sum(), ((g.eI == 0) & (g.eV == 0)).sum()]], float)
        if tb.sum() >= 2:
            tabs.append(tb)                     # no +0.5: it biases the pooled OR toward 1
    try:
        st = StratifiedTable(tabs); mh, mh_p = float(st.oddsratio_pooled), float(st.test_null_odds().pvalue)
    except Exception:
        mh, mh_p = np.nan, np.nan
    r = E.groupby(level=0).mean()
    rng = np.random.default_rng(seed)
    ci_seq = _boot_corr(r, np.array(r.index), B, rng)
    ci_ses = _boot_corr(r, np.array([session(t) for t in r.index]), B, rng)
    sr = r.assign(sess=[session(t) for t in r.index]).groupby("sess").mean()
    rho = r.eI.corr(r.eV)
    both, indep = float(((E.eI == 1) & (E.eV == 1)).mean()), float(E.eI.mean() * E.eV.mean())
    return (f"{label}: miss IR {E.eI.mean():.3f}, visible {E.eV.mean():.3f}; both {both:.3f} vs {indep:.3f} if independent "
            f"| within-seq MH OR {mh:.2f} (p={mh_p:.2g}) | per-seq corr {rho:.2f} "
            f"[seq bootstrap {ci_seq[0]:.2f}, {ci_seq[1]:.2f}; session bootstrap {ci_ses[0]:.2f}, {ci_ses[1]:.2f}] "
            f"| {len(r)} seqs, {len(sr)} sessions, per-session corr {sr.eI.corr(sr.eV):.2f}"), \
        dict(label=label, miss_ir=float(E.eI.mean()), miss_vis=float(E.eV.mean()), both=both, indep=indep,
             mh_or=mh, mh_p=mh_p, corr=float(rho), ci_seq=list(ci_seq), ci_session=list(ci_ses),
             seqs=int(len(r)), sessions=int(len(sr)))


# =====================================================================
# pipeline
# =====================================================================
def main():
    for pkg in ("ultralytics", "statsmodels", "huggingface_hub", "opencv-python-headless"):
        mod = {"opencv-python-headless": "cv2"}.get(pkg, pkg)
        try:
            __import__(mod)
        except ImportError:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=True)
    import cv2
    from huggingface_hub import list_repo_files, hf_hub_download

    DRIVE = "/content/drive"
    if not os.path.isdir(f"{DRIVE}/MyDrive"):
        try:
            from google.colab import drive
            drive.mount(DRIVE)
        except Exception as e:
            sys.exit(f"Could not mount Google Drive ({e}). Run in a cell first:\n"
                     "  from google.colab import drive; drive.mount('/content/drive')")
    WORK = f"{DRIVE}/MyDrive/counter_uas_attack"
    os.makedirs(WORK, exist_ok=True); os.chdir(WORK)
    print("working directory:", WORK, flush=True)

    NTR = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    NTE = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    EPOCHS = int(sys.argv[3]) if len(sys.argv) > 3 else 25
    SPREAD = len(sys.argv) > 4 and sys.argv[4] == "spread"; SUF = "_spread" if SPREAD else ""
    lines, summary = [], {}

    def say(x):
        print(x, flush=True); lines.append(str(x))
        open(f"attack_results{SUF}.txt", "w").write("\n".join(lines))

    def record(key, res):
        txt, d = res; say(txt); summary[key] = d
        json.dump(summary, open(f"attack_results{SUF}.json", "w"), indent=1, default=float)

    # ---------- annotations (cached on Drive)
    files = list_repo_files(REPO, repo_type="dataset")
    seqs = {}
    for f in files:
        p = f.split("/")
        if len(p) >= 3 and p[-1] in ("infrared.mp4", "visible.mp4", "infrared.json", "visible.json"):
            seqs.setdefault("/".join(p[:-1]), set()).add(p[-1])
    full = sorted(s for s, v in seqs.items() if len(v) == 4)
    ANN = json.load(open("annotations.json")) if os.path.exists("annotations.json") else {}
    new = 0
    for s in full:
        if s in ANN:
            continue
        try:
            ANN[s] = {m: json.load(open(hf_hub_download(REPO, f"{s}/{m}.json", repo_type="dataset"))) for m in SENS}
            new += 1
        except Exception:
            pass
    if new:
        json.dump(ANN, open("annotations.json", "w"))

    def ok(a, m, f):
        return f < len(a[m]["exist"]) and a[m]["exist"][f] and len(a[m]["gt_rect"][f]) == 4 \
            and a[m]["gt_rect"][f][2] * a[m]["gt_rect"][f][3] > 0

    def frames(s, step):
        a = ANN[s]; n = min(len(a["infrared"]["exist"]), len(a["visible"]["exist"]))
        return [f for f in range(0, n, step) if ok(a, "infrared", f) and ok(a, "visible", f)]

    diag = lambda s, f: math.hypot(*ANN[s]["infrared"]["gt_rect"][f][2:])

    # ---------- split (frozen on Drive so every resume uses the same sequences)
    if os.path.exists("split.json"):
        SP = json.load(open("split.json"))
    else:
        far = [s for s in full if s in ANN and sum(diag(s, f) <= FAR_PX for f in frames(s, 25)) >= 2]
        fset = set(far)
        test = [s for s in full if s.startswith("test/") and s in ANN and s not in fset][:NTE]
        held = set(map(session, test)) | set(map(session, far))
        cand = [s for s in full if s.startswith("train/") and s in ANN and s not in fset]
        clean = [s for s in cand if session(s) not in held]
        train = clean[:NTR] + [s for s in cand if s not in clean][:max(0, NTR - len(clean))]
        nval = max(4, len(train) // 8)
        SP = dict(train=train[:-nval], val=train[-nval:], test=test, far=far)
        json.dump(SP, open("split.json", "w"), indent=1)
    if SPREAD:  # test sequences spread over many sessions, none used in training
        if "test_spread" not in SP:
            used = set(map(session, SP["train"] + SP["val"])); fset = set(SP["far"]); by = {}
            for s in full:
                if s.startswith("test/") and s in ANN and s not in fset and session(s) not in used and frames(s, 50):
                    by.setdefault(session(s), []).append(s)
            ses = sorted(k for k, v in by.items() if len(v) >= 2)
            idx = np.linspace(0, len(ses) - 1, min(math.ceil(NTE / 3), len(ses))).round().astype(int)
            SP["test_spread"] = [x for j in dict.fromkeys(idx.tolist()) for x in by[ses[j]][:3]][:NTE]
            json.dump(SP, open("split.json", "w"), indent=1)
        SP["test"] = SP["test_spread"]
    trs = set(map(session, SP["train"] + SP["val"]))
    say(f"sequences: {len(full)} | train {len(SP['train'])}, val {len(SP['val'])}, test {len(SP['test'])}, "
        f"far-range {len(SP['far'])} ({len(set(map(session, SP['far'])))} sessions)")
    say(f"training sessions also in test: {len(trs & set(map(session, SP['test'])))} of {len(set(map(session, SP['test'])))} "
        f"| also in far-range: {len(trs & set(map(session, SP['far'])))} of {len(set(map(session, SP['far'])))}")

    # ---------- frame extraction (images + YOLO labels, on Drive)
    def extract(seq_list, step, root):
        os.makedirs(f"{root}/images", exist_ok=True); os.makedirs(f"{root}/labels", exist_ok=True)
        meta = []
        for k, s in enumerate(seq_list):
            want = set(frames(s, step))
            if not want:
                continue
            for m in SENS:
                tag = f"{s.replace('/', '_')}__{m}"
                lab = lambda p: p.replace("/images/", "/labels/").replace(".jpg", ".txt")
                done = [p for p in glob.glob(f"{root}/images/{tag}__*.jpg") if os.path.exists(lab(p))]
                if len(done) < len(want):
                    vid = hf_hub_download(REPO, f"{s}/{m}.mp4", repo_type="dataset")
                    cap = cv2.VideoCapture(vid); f = 0; todo = set(want)
                    while todo:
                        good, img = cap.read()
                        if not good:
                            break
                        if f in todo:
                            H, W = img.shape[:2]; x, y, w, h = ANN[s][m]["gt_rect"][f]
                            p = f"{root}/images/{tag}__{f}.jpg"; cv2.imwrite(p, img)
                            open(lab(p), "w").write(f"0 {(x + w / 2) / W:.6f} {(y + h / 2) / H:.6f} {w / W:.6f} {h / H:.6f}\n")
                            todo.discard(f)
                        f += 1
                    cap.release()
                    try:
                        os.remove(os.path.realpath(vid))
                    except Exception:
                        pass
                    done = [p for p in glob.glob(f"{root}/images/{tag}__*.jpg") if os.path.exists(lab(p))]
                for p in done:
                    f = int(p.rsplit("__", 1)[1][:-4]); x, y, w, h = ANN[s][m]["gt_rect"][f]
                    meta.append(dict(img=p, track=s, session=session(s), frame=f, sensor=m,
                                     x=x, y=y, w=w, h=h, diag=diag(s, f)))
            if k % 10 == 0:
                print(root, k, "/", len(seq_list), len(meta), "images", flush=True)
        return pd.DataFrame(meta)

    TR = extract(SP["train"], 50, "yolo/train")
    VA = extract(SP["val"], 50, "yolo/val")
    TE = extract(SP["test"], 50, "eval/test")
    FA = extract(SP["far"], 25, "eval/far")
    say(f"images: train {len(TR)}, val {len(VA)}, test {len(TE)}, far {len(FA)} "
        f"(of which {int((FA.diag <= FAR_PX).sum()) if len(FA) else 0} truly far)")

    # ---------- train YOLO11n from a local copy (Drive is slow for many small files)
    from ultralytics import YOLO
    LOCAL = "/content/yolo_local"
    if not os.path.exists(f"{LOCAL}/.copied"):
        print("copying training images to local disk...", flush=True)
        shutil.copytree("yolo", LOCAL, dirs_exist_ok=True); open(f"{LOCAL}/.copied", "w").write("ok")
    open("drone.yaml", "w").write(f"path: {LOCAL}\ntrain: train/images\nval: val/images\nnames:\n  0: drone\n")
    run_dir = os.path.abspath("runs/detect/drone")
    best, last, rc = f"{run_dir}/weights/best.pt", f"{run_dir}/weights/last.pt", f"{run_dir}/results.csv"
    epochs_done = lambda: len(pd.read_csv(rc)) if os.path.exists(rc) else 0
    if epochs_done() < EPOCHS:
        if os.path.exists(last):
            print(f"resuming YOLO training at epoch {epochs_done()} of {EPOCHS}", flush=True)
            try:
                YOLO(last).train(resume=True)
            except Exception:
                if epochs_done() < EPOCHS:
                    raise                                   # a real failure, not a finished run
        else:
            YOLO("yolo11n.pt").train(data=os.path.abspath("drone.yaml"), epochs=EPOCHS, imgsz=640, batch=32,
                                     project=os.path.abspath("runs/detect"), name="drone", exist_ok=True,
                                     verbose=False, plots=False, workers=2)
    if epochs_done() < EPOCHS:
        sys.exit(f"training stopped at epoch {epochs_done()} of {EPOCHS}; run the cell again to resume")
    model = YOLO(best if os.path.exists(best) else last)
    try:
        say(f"detector validation mAP50: {pd.read_csv(rc).filter(like='mAP50(B)').iloc[-1, 0]:.3f}")
    except Exception:
        pass

    # ---------- evaluation (cached)
    def evaluate(D):
        hits = []
        for i in range(0, len(D), 32):
            chunk = D.iloc[i:i + 32]
            res = model.predict(list(chunk.img), imgsz=640, conf=0.25, verbose=False)
            for (_, r), out in zip(chunk.iterrows(), res):
                b = out.boxes.xyxy.cpu().numpy() if out.boxes is not None else np.zeros((0, 4))
                cx, cy = r.x + r.w / 2, r.y + r.h / 2; tol = max(r.w, r.h, 4)
                hits.append(int(any(abs((b0 + b2) / 2 - cx) <= tol and abs((b1 + b3) / 2 - cy) <= tol
                                    for b0, b1, b2, b3 in b)))
        D = D.copy(); D["err"] = 1 - np.array(hits); D["size"] = np.sqrt(D.w * D.h)
        return D
    if os.path.exists(f"attack_votes{SUF}.csv"):
        V = pd.read_csv(f"attack_votes{SUF}.csv", dtype={"track": str, "session": str})
        TE, FA = V[V.split == "test"].copy(), V[V.split == "far"].copy()
    else:
        TE, FA = evaluate(TE), evaluate(FA)
        pd.concat([TE.assign(split="test"), FA.assign(split="far")]).to_csv(f"attack_votes{SUF}.csv", index=False)
    FF = FA[FA.diag <= FAR_PX]
    say(f"detector evaluated: {len(TE)} test, {len(FA)} far-sequence, {len(FF)} truly-far frames")

    # ---------- analyses
    sets = [("TEST", TE, 3), ("FAR SEQUENCES (all frames)", FA, 3), (f"FAR FRAMES (IR diag <= {FAR_PX} px)", FF, 2)]
    say("\n=== 1. Within-sequence clustering of YOLO11n misses ===")
    for nm, D, mn in sets:
        for m in SENS:
            record(f"tau|{nm}|{m}", tau_line(D[D.sensor == m], f"{nm} {m}", mn))
    say("\n=== 2. Cross-sensor dependence (does IR miss when visible misses?) ===")
    for nm, D, _ in sets:
        record(f"cross|{nm}", cross_line(D, nm))
    say("\n=== 3. Real session labels: sequence vs session clustering ===")
    for nm, D, mn in [("TEST", TE, 3), ("FAR SEQUENCES", FA, 3), ("ALL", pd.concat([TE, FA]), 3)]:
        for m in SENS:
            record(f"session|{nm}|{m}", session_line(D[D.sensor == m], f"{nm} {m}", mn))
    say("\nDONE")


if __name__ == "__main__":
    main()
