"""
Follow-up experiments for the counter-UAS paper. Run in Colab after attack.py:
    from google.colab import drive; drive.mount('/content/drive')
    %run /content/drive/MyDrive/counter_uas_attack/extra.py
Stages (each resumable and independent; results in extra_results.txt):
  1. POOLED: Mantel-Haenszel odds ratio pooled over both test sets, with CI and
     a Breslow-Day test of homogeneity across sequences.
  2. VIS1280: visible test-B frames re-evaluated at 1280 px (resolution confound).
  3. AUDIT: frames with no drone in either sensor plus drone frames, each with a
     temporal view (same sensor, 12 frames later) -> audit_votes.csv for a real audit.
  4. SEEDS: YOLO11n retrained with seeds 1 and 2 (needs a GPU; skipped on CPU).
  5. SCENES: Halmstad first-frame scene groups (Ward, K = 10/20/40) -> halmstad_scenes.csv
"""
import os, sys, json, glob, shutil, subprocess, traceback
import numpy as np, pandas as pd

WORK = "/content/drive/MyDrive/counter_uas_attack"
if not os.path.isdir(WORK):
    from google.colab import drive; drive.mount("/content/drive")
os.chdir(WORK); sys.path.insert(0, WORK)
import attack as A
import cv2
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
from statsmodels.stats.contingency_tables import StratifiedTable

EPOCHS, OFFSET = 25, 12
lines = open("extra_results.txt").read().split("\n") if os.path.exists("extra_results.txt") else []
def say(x):
    print(x, flush=True); lines.append(str(x)); open("extra_results.txt", "w").write("\n".join(lines))
def done(tag):
    return f"[{tag}] DONE" in lines

ANN = json.load(open("annotations.json")); SP = json.load(open("split.json"))
VA = pd.read_csv("attack_votes.csv", dtype={"track": str, "session": str})
VB = pd.read_csv("attack_votes_spread.csv", dtype={"track": str, "session": str})
TA, TB = VA[VA.split == "test"].copy(), VB[VB.split == "test"].copy()
BEST = "runs/detect/drone/weights/best.pt"

def hit_eval(model, D, imgsz=640):
    out = []
    for i in range(0, len(D), 32):
        ch = D.iloc[i:i + 32]
        for (_, r), res in zip(ch.iterrows(), model.predict(list(ch.img), imgsz=imgsz, conf=0.25, verbose=False)):
            b = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else np.zeros((0, 4))
            cx, cy = r.x + r.w / 2, r.y + r.h / 2; tol = max(r.w, r.h, 4)
            out.append(int(any(abs((b0 + b2) / 2 - cx) <= tol and abs((b1 + b3) / 2 - cy) <= tol for b0, b1, b2, b3 in b)))
    D = D.copy(); D["err"] = 1 - np.array(out); return D

def tables(D):
    P = D.pivot_table(index=["track", "frame"], columns="sensor", values="err").dropna().astype(int)
    T = []
    for _, g in P.groupby(level=0):
        tb = np.array([[((g.infrared == 1) & (g.visible == 1)).sum(), ((g.infrared == 1) & (g.visible == 0)).sum()],
                       [((g.infrared == 0) & (g.visible == 1)).sum(), ((g.infrared == 0) & (g.visible == 0)).sum()]], float)
        if tb.sum() >= 2: T.append(tb)
    return T

def stage(tag, fn):
    if done(tag): print(f"{tag}: already done"); return
    try:
        if fn() is not False: say(f"[{tag}] DONE")
    except Exception as e: say(f"[{tag}] FAILED (will retry on rerun): {e!r}"); traceback.print_exc()

# ---------------------------------------------------------------- 1. pooled odds ratio
def pooled():
    a, b = set(TA.track), set(TB.track)
    U = pd.concat([TA, TB]).drop_duplicates(["track", "frame", "sensor"])
    for nm, D in (("test A", TA), ("test B", TB), ("A+B unique", U)):
        st = StratifiedTable(tables(D)); lo, hi = st.oddsratio_pooled_confint()
        bd = st.test_equal_odds(adjust=True)
        say(f"[POOLED] {nm}: MH OR {st.oddsratio_pooled:.2f} [{lo:.2f}, {hi:.2f}] p={st.test_null_odds().pvalue:.2g} "
            f"| Breslow-Day homogeneity p={bd.pvalue:.2g} | {D.track.nunique()} seqs")
    say(f"[POOLED] sequences shared by A and B: {len(a & b)} of {len(a)} and {len(b)}")
stage("POOLED", pooled)

# ---------------------------------------------------------------- 2. visible at 1280 px
def vis1280():
    m = YOLO(BEST); V = TB[TB.sensor == "visible"]
    if os.path.exists("vis1280_votes.csv"): V2 = pd.read_csv("vis1280_votes.csv", dtype={"track": str, "session": str})
    else: V2 = hit_eval(m, V, 1280); V2.to_csv("vis1280_votes.csv", index=False)
    say(f"[VIS1280] visible recall at 640 px {1 - V.err.mean():.3f} -> at 1280 px {1 - V2.err.mean():.3f}")
    say("[VIS1280] " + A.tau_line(V2, "visible @1280")[0])
    say("[VIS1280] " + A.cross_line(pd.concat([TB[TB.sensor == "infrared"], V2]), "IR@640 x visible@1280")[0])
stage("VIS1280", vis1280)

# ---------------------------------------------------------------- 3. audit votes (negatives + temporal view)
def audit():
    if os.path.exists("audit_votes.csv"):
        return
    m = YOLO(BEST); root = "eval/audit/images"; os.makedirs(root, exist_ok=True)
    ex = lambda s, s_, f: f < len(ANN[s][s_]["exist"]) and ANN[s][s_]["exist"][f]
    rows = []
    for k, s in enumerate(SP["test_spread"]):
        n = min(len(ANN[s]["infrared"]["exist"]), len(ANN[s]["visible"]["exist"])) - OFFSET
        st = lambda f: (ex(s, "infrared", f), ex(s, "visible", f))
        pos = [f for f in A_frames(s, 50) if f < n and st(f) == (1, 1) and st(f + OFFSET) == (1, 1)]
        neg = [f for f in range(0, n, 25) if st(f) == (0, 0) and st(f + OFFSET) == (0, 0)][:15]
        want = {f for f in pos + neg} | {f + OFFSET for f in pos + neg}
        if not want: continue
        paths = {}
        for sen in A.SENS:
            vid = hf_hub_download(A.REPO, f"{s}/{sen}.mp4", repo_type="dataset"); cap = cv2.VideoCapture(vid); f = 0; todo = set(want)
            while todo:
                ok, img = cap.read()
                if not ok: break
                if f in todo:
                    p = f"{root}/{s.replace('/', '_')}__{sen}__{f}.jpg"; cv2.imwrite(p, img); paths[(sen, f)] = p; todo.discard(f)
                f += 1
            cap.release()
            try: os.remove(os.path.realpath(vid))
            except Exception: pass
        for f in pos + neg:
            if all((sen, g) in paths for sen in A.SENS for g in (f, f + OFFSET)):
                rows.append(dict(track=s, session=A.session(s), frame=f, truth=int(f in pos),
                                 **{f"{sen[:3]}{suf}": paths[(sen, f + d)] for sen in A.SENS for suf, d in (("", 0), ("_t", OFFSET))}))
        print("audit", k, "/", len(SP["test_spread"]), len(rows), flush=True)
    R = pd.DataFrame(rows); cols = ["inf", "vis", "inf_t", "vis_t"]
    for c in cols:
        res = []
        for i in range(0, len(R), 32):
            res += [int(r.boxes is not None and len(r.boxes) > 0) for r in m.predict(list(R[c].iloc[i:i + 32]), imgsz=640, conf=0.25, verbose=False)]
        R[c] = res
    R[["track", "session", "frame", "truth"] + cols].to_csv("audit_votes.csv", index=False)
    say(f"[AUDIT] {len(R)} frames ({int(R.truth.sum())} drone, {int((1 - R.truth).sum())} no drone) -> audit_votes.csv")
    for c in cols:
        say(f"[AUDIT] {c}: sensitivity {R.loc[R.truth == 1, c].mean():.3f}, specificity {1 - R.loc[R.truth == 0, c].mean():.3f}")

def A_frames(s, step):
    a = ANN[s]; n = min(len(a["infrared"]["exist"]), len(a["visible"]["exist"]))
    okf = lambda m, f: a[m]["exist"][f] and len(a[m]["gt_rect"][f]) == 4 and a[m]["gt_rect"][f][2] * a[m]["gt_rect"][f][3] > 0
    return [f for f in range(0, n, step) if okf("infrared", f) and okf("visible", f)]
stage("AUDIT", audit)

# ---------------------------------------------------------------- 4. extra seeds
def seeds():
    import torch
    if not torch.cuda.is_available():
        print("SEEDS: no GPU in this runtime; rerun on a T4 to do this stage", flush=True); return False
    LOCAL = "/content/yolo_local"
    if not os.path.exists(f"{LOCAL}/.copied"):
        shutil.copytree("yolo", LOCAL, dirs_exist_ok=True); open(f"{LOCAL}/.copied", "w").write("ok")
    for s in (1, 2):
        if os.path.exists(f"seed{s}_votes.csv"): continue
        rd = os.path.abspath(f"runs/detect/drone_s{s}"); rc = f"{rd}/results.csv"
        n = lambda: len(pd.read_csv(rc)) if os.path.exists(rc) else 0
        if n() < EPOCHS:
            if os.path.exists(f"{rd}/weights/last.pt"):
                try: YOLO(f"{rd}/weights/last.pt").train(resume=True)
                except Exception:
                    if n() < EPOCHS: raise
            else:
                YOLO("yolo11n.pt").train(data=os.path.abspath("drone.yaml"), epochs=EPOCHS, imgsz=640, batch=32, seed=s,
                                         project=os.path.abspath("runs/detect"), name=f"drone_s{s}", exist_ok=True,
                                         verbose=False, plots=False, workers=2)
        D = hit_eval(YOLO(f"{rd}/weights/best.pt"), TB); D.to_csv(f"seed{s}_votes.csv", index=False)
    for s in (0, 1, 2):
        D = TB if s == 0 else pd.read_csv(f"seed{s}_votes.csv", dtype={"track": str, "session": str})
        for sen in A.SENS: say(f"[SEEDS] seed {s} " + A.tau_line(D[D.sensor == sen], f"test B {sen}")[0])
        st = StratifiedTable(tables(D)); lo, hi = st.oddsratio_pooled_confint()
        say(f"[SEEDS] seed {s} within-seq MH OR {st.oddsratio_pooled:.2f} [{lo:.2f}, {hi:.2f}]")
stage("SEEDS", seeds)

# ---------------------------------------------------------------- 5. Halmstad scene groups
def scenes():
    from scipy.cluster.hierarchy import linkage, fcluster
    H = "/content/halmstad"
    if not os.path.isdir(H):
        subprocess.run(["git", "clone", "-q", "--depth", "1", "--filter=blob:none", "--sparse",
                        "https://github.com/DroneDetectionThesis/Drone-detection-dataset.git", H], check=True)
        subprocess.run(["git", "-C", H, "sparse-checkout", "set", "--no-cone", "*_DRONE_*.mp4", "*_BIRD_*.mp4"], check=True)
    vids = sorted(glob.glob(f"{H}/**/*_DRONE_*.mp4", recursive=True) + glob.glob(f"{H}/**/*_BIRD_*.mp4", recursive=True)); rows = []
    if not vids: raise RuntimeError("no Halmstad videos found after sparse checkout")
    for v in vids:
        cap = cv2.VideoCapture(v); ok, img = cap.read(); cap.release()
        if not ok: continue
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thumb = cv2.resize(g, (16, 12), interpolation=cv2.INTER_AREA).ravel() / 255.0
        hist = np.histogram(g, bins=16, range=(0, 255))[0] / g.size
        rows.append(dict(track=os.path.basename(v)[:-4], sensor="IR" if os.path.basename(v).startswith("IR_") else "V", **{f"f{i}": x for i, x in enumerate(np.r_[thumb, hist])}))
    F = pd.DataFrame(rows); out = []
    for sen, g in F.groupby("sensor"):
        X = g.filter(like="f").values; X = (X - X.mean(0)) / (X.std(0) + 1e-9); Z = linkage(X, "ward")
        out.append(g[["track", "sensor"]].assign(**{f"K{k}": fcluster(Z, k, "maxclust") for k in (10, 20, 40)}))
    pd.concat(out).to_csv("halmstad_scenes.csv", index=False)
    say(f"[SCENES] {len(F)} Halmstad videos grouped -> halmstad_scenes.csv")
stage("SCENES", scenes)
say("[END] extra.py finished")
