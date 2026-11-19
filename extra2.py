"""
Round 2 experiments (run after attack.py and extra.py, on a T4):
    !pip -q install ultralytics statsmodels huggingface_hub
    from google.colab import drive; drive.mount('/content/drive')
    %run /content/drive/MyDrive/counter_uas_attack/extra2.py
Stages (resumable; results appended to extra2_results.txt):
  AVID   : list the AVID paired RGB-IR bird/drone dataset on Hugging Face -> avid_listing.txt
  ARCH   : train YOLO11s and RT-DETR-l on the same Anti-UAV split, evaluate test set B,
           report tau per sensor and the within-sequence cross-sensor odds ratio.
"""
import os, sys, json, shutil, traceback
import numpy as np, pandas as pd
WORK = "/content/drive/MyDrive/counter_uas_attack"
os.chdir(WORK); sys.path.insert(0, WORK)
import attack as A
from ultralytics import YOLO, RTDETR
from huggingface_hub import list_repo_files, hf_hub_download
from statsmodels.stats.contingency_tables import StratifiedTable

EPOCHS = 25
OUT = "extra2_results.txt"
lines = open(OUT).read().split("\n") if os.path.exists(OUT) else []
def say(x):
    print(x, flush=True); lines.append(str(x)); open(OUT, "w").write("\n".join(lines))
def stage(tag, fn):
    if f"[{tag}] DONE" in lines: print(tag, "already done"); return
    try:
        if fn() is not False: say(f"[{tag}] DONE")
    except Exception as e: say(f"[{tag}] FAILED (will retry on rerun): {e!r}"); traceback.print_exc()

VB = pd.read_csv("attack_votes_spread.csv", dtype={"track": str, "session": str}); TB = VB[VB.split == "test"].copy()

def hit_eval(model, D, imgsz=640):
    out = []
    for i in range(0, len(D), 16):
        ch = D.iloc[i:i + 16]
        for (_, r), res in zip(ch.iterrows(), model.predict(list(ch.img), imgsz=imgsz, conf=0.25, verbose=False)):
            b = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else np.zeros((0, 4))
            cx, cy = r.x + r.w / 2, r.y + r.h / 2; tol = max(r.w, r.h, 4)
            out.append(int(any(abs((b0 + b2) / 2 - cx) <= tol and abs((b1 + b3) / 2 - cy) <= tol for b0, b1, b2, b3 in b)))
    D = D.copy(); D["err"] = 1 - np.array(out); return D

def mh(D):
    P = D.pivot_table(index=["track", "frame"], columns="sensor", values="err").dropna().astype(int); T = []
    for _, g in P.groupby(level=0):
        tb = np.array([[((g.infrared == 1) & (g.visible == 1)).sum(), ((g.infrared == 1) & (g.visible == 0)).sum()],
                       [((g.infrared == 0) & (g.visible == 1)).sum(), ((g.infrared == 0) & (g.visible == 0)).sum()]], float)
        if tb.sum() >= 2: T.append(tb)
    st = StratifiedTable(T); lo, hi = st.oddsratio_pooled_confint(); return st.oddsratio_pooled, lo, hi

# ------------------------------------------------------------ AVID listing
def avid():
    files = list_repo_files("Dharini24/AVID", repo_type="dataset")
    ext = pd.Series([os.path.splitext(f)[1].lower() for f in files]).value_counts()
    tops = pd.Series([f.split("/")[0] for f in files]).value_counts()
    depth2 = pd.Series(["/".join(f.split("/")[:2]) for f in files if "/" in f]).value_counts().head(40)
    txt = [f"files: {len(files)}", "extensions:", ext.to_string(), "top level:", tops.to_string(), "first two levels:", depth2.to_string(),
           "sample paths:"] + files[:25] + files[-15:]
    for f in files:
        if f.lower().endswith((".md", ".json", ".yaml", ".yml", ".txt", ".csv")) and f.count("/") <= 1:
            try: txt += [f"----- {f}", open(hf_hub_download("Dharini24/AVID", f, repo_type="dataset")).read()[:3000]]
            except Exception as e: txt.append(f"(could not read {f}: {e})")
    open("avid_listing.txt", "w").write("\n".join(map(str, txt)))
    say(f"[AVID] {len(files)} files; top-level entries: {', '.join(tops.index[:8])} -> avid_listing.txt")
stage("AVID", avid)

# ------------------------------------------------------------ new architectures
def arch():
    import torch
    if not torch.cuda.is_available():
        print("ARCH: needs a GPU runtime", flush=True); return False
    LOCAL = "/content/yolo_local"
    if not os.path.exists(f"{LOCAL}/.copied"):
        shutil.copytree("yolo", LOCAL, dirs_exist_ok=True); open(f"{LOCAL}/.copied", "w").write("ok")
    for name, ctor, weights, batch in (("yolo11s", YOLO, "yolo11s.pt", 32), ("rtdetr_l", RTDETR, "rtdetr-l.pt", 8)):
        vf = f"votes_{name}.csv"
        if not os.path.exists(vf):
            rd = os.path.abspath(f"runs/detect/{name}"); rc = f"{rd}/results.csv"
            n = lambda: len(pd.read_csv(rc)) if os.path.exists(rc) else 0
            if n() < EPOCHS:
                if os.path.exists(f"{rd}/weights/last.pt"):
                    try: ctor(f"{rd}/weights/last.pt").train(resume=True)
                    except Exception:
                        if n() < EPOCHS: raise
                else:
                    ctor(weights).train(data=os.path.abspath("drone.yaml"), epochs=EPOCHS, imgsz=640, batch=batch,
                                        project=os.path.abspath("runs/detect"), name=name, exist_ok=True,
                                        verbose=False, plots=False, workers=2)
            hit_eval(ctor(f"{rd}/weights/best.pt"), TB).to_csv(vf, index=False)
        D = pd.read_csv(vf, dtype={"track": str, "session": str})
        for sen in A.SENS: say(f"[ARCH] {name} " + A.tau_line(D[D.sensor == sen], f"test B {sen}")[0])
        o, lo, hi = mh(D); say(f"[ARCH] {name} within-seq MH OR {o:.2f} [{lo:.2f}, {hi:.2f}]")
        try:
            r = pd.read_csv(f"runs/detect/{name}/results.csv"); say(f"[ARCH] {name} val mAP50 {r.filter(like='mAP50(B)').iloc[-1, 0]:.3f}")
        except Exception: pass
stage("ARCH", arch)
say("[END] extra2.py finished")
