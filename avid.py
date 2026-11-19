"""
AVID experiment (paired RGB-IR, bird and drone videos; CC BY-NC 4.0). Run in Colab on a T4:
    !pip -q install ultralytics statsmodels huggingface_hub
    from google.colab import drive; drive.mount('/content/drive')
    %run /content/drive/MyDrive/counter_uas_attack/avid.py
Images live on local disk (Drive is nearly full); checkpoints and results go to Drive.
Stages:
  DATA  : download 40 train videos + all 13 val videos; build a 2-class YOLO set (every 4th train frame).
  TRAIN : YOLO11n, 20 epochs (resumable from Drive).
  SCORE : per-frame max drone confidence in each sensor on all val frames (and 12 frames later).
  ANALYZE: (a) sensitivity/specificity, clustering, cross-sensor dependence with real negatives;
           (b) Dawid-Skene audit vs ground truth, now including specificity;
           (c) fusion: independent (naive Bayes) vs learned stacking vs interaction stacking, frame level;
               and temporal fusion over 25-frame segments: independent vs design-effect-corrected vs mean.
"""
import os, sys, json, glob, shutil, random, traceback
import numpy as np, pandas as pd
WORK = "/content/drive/MyDrive/counter_uas_attack"; LOC = "/content/avid"; os.makedirs(LOC, exist_ok=True)
sys.path.insert(0, WORK); import attack as A
from huggingface_hub import list_repo_files, snapshot_download, hf_hub_download
from ultralytics import YOLO
REPO, EPOCHS, OFF, SEG = "Dharini24/AVID", 20, 12, 25
OUT = f"{WORK}/avid_results.txt"
lines = open(OUT).read().split("\n") if os.path.exists(OUT) else []
def say(x):
    print(x, flush=True); lines.append(str(x)); open(OUT, "w").write("\n".join(lines))
def stage(tag, fn):
    if f"[{tag}] DONE" in lines: print(tag, "done"); return
    try:
        if fn() is not False: say(f"[{tag}] DONE")
    except Exception as e: say(f"[{tag}] FAILED (rerun retries): {e!r}"); traceback.print_exc(); raise

def jl(f): return [json.loads(l) for l in open(hf_hub_download(REPO, f, repo_type="dataset")) if l.strip()]
FILES = list_repo_files(REPO, repo_type="dataset")
vids_in = lambda split: sorted({f.split("/")[2] for f in FILES if f.startswith(f"labels/{split}/") and f.count("/") >= 3})
rng = random.Random(0)
train_vids = vids_in("train"); rng.shuffle(train_vids)
bird = [v for v in train_vids if v.startswith("BIRD")][:20]; drone = [v for v in train_vids if v.startswith("DRONE")][:20]
TRV, VAV = sorted(bird + drone), vids_in("val")

def data():
    pats = [f"{d}/{m}/train/{v}/*" for v in TRV for d, m in (("images", "ir"), ("images", "rgb"))] + [f"labels/train/{v}/*" for v in TRV]
    pats += [f"images/*/val/*", "labels/val/*", "metadata/pairs_val.jsonl"]
    snapshot_download(REPO, repo_type="dataset", local_dir=f"{LOC}/raw", allow_patterns=pats, max_workers=16)
    for split, vids, step in (("train", TRV, 4), ("val", VAV, 8)):
        os.makedirs(f"{LOC}/yolo/{split}/images", exist_ok=True); os.makedirs(f"{LOC}/yolo/{split}/labels", exist_ok=True)
        for v in vids:
            for lab in sorted(glob.glob(f"{LOC}/raw/labels/{'train' if split == 'train' else 'val'}/{v}/*.txt"))[::step]:
                fr = os.path.basename(lab)[:-4]
                for m in ("ir", "rgb"):
                    im = f"{LOC}/raw/images/{m}/{'train' if split == 'train' else 'val'}/{v}/{fr}.jpg"
                    if os.path.exists(im):
                        shutil.copy(im, f"{LOC}/yolo/{split}/images/{v}_{m}_{fr}.jpg"); shutil.copy(lab, f"{LOC}/yolo/{split}/labels/{v}_{m}_{fr}.txt")
    open(f"{LOC}/avid.yaml", "w").write(f"path: {LOC}/yolo\ntrain: train/images\nval: val/images\nnames:\n  0: bird\n  1: drone\n")
    say(f"[DATA] train images {len(os.listdir(f'{LOC}/yolo/train/images'))}, val images {len(os.listdir(f'{LOC}/yolo/val/images'))}, "
        f"train videos {len(TRV)}, val videos {len(VAV)}")
if not os.path.exists(f"{LOC}/avid.yaml"):  # local disk is empty after a runtime reset: (re)build the data
    lines[:] = [l for l in lines if not l.startswith("[DATA]")]; stage("DATA", data)

RD = f"{WORK}/runs/avid_y11n"; RC = f"{RD}/results.csv"
def train():
    n = lambda: len(pd.read_csv(RC)) if os.path.exists(RC) else 0
    if n() >= EPOCHS: return
    if os.path.exists(f"{RD}/weights/last.pt"):
        try: YOLO(f"{RD}/weights/last.pt").train(resume=True)
        except Exception:
            if n() < EPOCHS: raise
    else:
        os.chdir(LOC)
        YOLO("yolo11n.pt").train(data=f"{LOC}/avid.yaml", epochs=EPOCHS, imgsz=640, batch=32, project=f"{WORK}/runs",
                                 name="avid_y11n", exist_ok=True, verbose=False, plots=False, workers=4)
    r = pd.read_csv(RC); say(f"[TRAIN] val mAP50 {r.filter(like='mAP50(B)').iloc[-1, 0]:.3f}")
stage("TRAIN", train)

SC = f"{WORK}/avid_scores.csv"
def score():
    if os.path.exists(SC): return
    model = YOLO(f"{RD}/weights/best.pt"); P = jl("metadata/pairs_val.jsonl"); rows = []
    for r in P:
        v, f = r["video_id"], int(r["frame_idx"])
        rows.append(dict(video=v, frame=f, truth=int("drone" in r.get("classes_present", []))))
    D = pd.DataFrame(rows)
    for m in ("ir", "rgb"):
        for suf, d in (("", 0), ("_t", OFF)):
            paths = [f"{LOC}/raw/images/{m}/val/{v}/{f + d:06d}.jpg" for v, f in zip(D.video, D.frame)]
            ok = [os.path.exists(p) for p in paths]; s = np.full(len(D), np.nan)
            idx = [i for i, o in enumerate(ok) if o]
            for i in range(0, len(idx), 64):
                ch = idx[i:i + 64]
                for j, res in zip(ch, model.predict([paths[k] for k in ch], imgsz=640, conf=0.001, verbose=False)):
                    b = res.boxes; c = b.conf.cpu().numpy() if b is not None else np.zeros(0); k = b.cls.cpu().numpy() if b is not None else np.zeros(0)
                    s[j] = float(c[k == 1].max()) if (k == 1).any() else 0.0
            D[f"{m}{suf}"] = s
    D.to_csv(SC, index=False); say(f"[SCORE] {len(D)} val frame pairs scored ({int(D.truth.sum())} drone, {int((1 - D.truth).sum())} no drone)")
stage("SCORE", score)

def analyze():
    from sklearn.linear_model import LogisticRegression
    from statsmodels.stats.contingency_tables import StratifiedTable
    D = pd.read_csv(SC).dropna(subset=["ir", "rgb"]); D["seg"] = D.video + "_" + (D.frame // SEG).astype(str); D["session"] = D.video
    for c in ("ir", "rgb", "ir_t", "rgb_t"): D[c + "_v"] = (D[c] >= 0.25).astype(float).where(D[c].notna())
    # (a) operating characteristics, clustering, dependence
    for m in ("ir", "rgb"):
        se, sp = D.loc[D.truth == 1, m + "_v"].mean(), 1 - D.loc[D.truth == 0, m + "_v"].mean()
        say(f"[ANALYZE] {m}: sensitivity {se:.3f}, specificity {sp:.3f}")
        for cls, nm in ((1, "drone frames (miss)"), (0, "bird frames (false alarm)")):
            g = D[D.truth == cls].copy(); g["err"] = (g[m + "_v"] != cls).astype(int); g["size"] = 1.0; g["track"] = g.seg
            say("[ANALYZE] " + A.tau_line(g, f"{m} {nm}, 25-frame segments", 3)[0])
    for cls, nm in ((1, "drone frames"), (0, "bird frames")):
        g = D[D.truth == cls]; T = []
        for _, h in g.groupby("seg"):
            eI, eV = (h.ir_v != cls), (h.rgb_v != cls)
            tb = np.array([[(eI & eV).sum(), (eI & ~eV).sum()], [(~eI & eV).sum(), (~eI & ~eV).sum()]], float)
            if tb.sum() >= 2: T.append(tb)
        try:
            st = StratifiedTable(T); lo, hi = st.oddsratio_pooled_confint()
            say(f"[ANALYZE] cross-sensor error OR within segments, {nm}: {st.oddsratio_pooled:.2f} [{lo:.2f}, {hi:.2f}]")
        except Exception as e: say(f"[ANALYZE] cross-sensor OR {nm}: not estimable ({e})")
    # (b) Dawid-Skene vs truth with real negatives
    def ds(V):
        w = (V.mean(1) > .5) * .8 + .1
        for _ in range(2000):
            pi = w.mean(); a = ((w[:, None] * V).sum(0) + .5) / (w.sum() + 1); b = (((1 - w)[:, None] * V).sum(0) + .5) / ((1 - w).sum() + 1)
            L1 = np.log(pi) + (V * np.log(a) + (1 - V) * np.log(1 - a)).sum(1); L0 = np.log(1 - pi) + (V * np.log(b) + (1 - V) * np.log(1 - b)).sum(1)
            wn = 1 / (1 + np.exp(L0 - L1))
            if np.abs(wn - w).max() < 1e-9: break
            w = wn
        return w.mean(), a, 1 - b
    E = D.dropna(subset=["ir_t_v", "rgb_t_v"])
    for cols in (["ir_v", "rgb_v", "ir_t_v", "rgb_t_v"], ["ir_v", "rgb_v", "ir_t_v"]):
        pi, se, sp = ds(E[cols].values)
        tr = [(E.loc[E.truth == 1, c].mean(), 1 - E.loc[E.truth == 0, c].mean()) for c in cols]
        say(f"[AUDIT] DS {cols}: prevalence {pi:.3f} (true {E.truth.mean():.3f}) | " +
            "; ".join(f"{c} Se {s:.3f}/{t[0]:.3f} Sp {p:.3f}/{t[1]:.3f}" for c, s, p, t in zip(cols, se, sp, tr)) + "  (estimate/true)")
    # (c) fusion, leave-one-video-out
    lg = lambda s: np.log(np.clip(s, 1e-4, 1 - 1e-4) / (1 - np.clip(s, 1e-4, 1 - 1e-4)))
    D["zI"], D["zV"] = lg(D.ir), lg(D.rgb)
    def metrics(y, p):
        p = np.clip(p, 1e-6, 1 - 1e-6); bins = np.minimum((p * 10).astype(int), 9)
        ece = sum(abs(y[bins == k].mean() - p[bins == k].mean()) * (bins == k).mean() for k in range(10) if (bins == k).any())
        conf = np.maximum(p, 1 - p).mean(); acc = ((p > .5) == y).mean()
        return dict(acc=acc, brier=((p - y) ** 2).mean(), nll=-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean(), ece=ece, over=conf - acc)
    preds = {k: np.zeros(len(D)) for k in ("IR only", "RGB only", "independent (naive Bayes)", "stacked", "stacked + interaction")}
    seg_out = {k: [] for k in ("independent over time", "design-effect corrected", "segment mean")}
    for v in D.video.unique():
        tr, te = D[D.video != v], D[D.video == v]; ti, ei = tr.index, te.index
        f = lambda cols, X=None: LogisticRegression(C=1e3, max_iter=2000).fit(tr[cols].values, tr.truth.values)
        mI, mV = f(["zI"]), f(["zV"])
        pI, pV = mI.predict_proba(te[["zI"]].values)[:, 1], mV.predict_proba(te[["zV"]].values)[:, 1]
        prior = tr.truth.mean(); lp = np.log(prior / (1 - prior))
        nb = 1 / (1 + np.exp(-(lg(pI) + lg(pV) - lp)))
        X = lambda d: np.column_stack([d.zI, d.zV]); Xi = lambda d: np.column_stack([d.zI, d.zV, d.zI * d.zV])
        st = LogisticRegression(C=1e3, max_iter=2000).fit(X(tr), tr.truth); sti = LogisticRegression(C=1e3, max_iter=2000).fit(Xi(tr), tr.truth)
        preds["IR only"][D.index.get_indexer(ei)] = pI; preds["RGB only"][D.index.get_indexer(ei)] = pV
        preds["independent (naive Bayes)"][D.index.get_indexer(ei)] = nb
        preds["stacked"][D.index.get_indexer(ei)] = st.predict_proba(X(te))[:, 1]
        preds["stacked + interaction"][D.index.get_indexer(ei)] = sti.predict_proba(Xi(te))[:, 1]
        # temporal fusion over segments, using the stacked frame model; ICC of frame LLRs from training videos
        llr_tr = lg(st.predict_proba(X(tr))[:, 1]) - lp; llr_te = lg(st.predict_proba(X(te))[:, 1]) - lp
        g = pd.DataFrame(dict(seg=tr.seg.values, y=tr.truth.values, l=llr_tr)); iccs = []
        for c in (0, 1):
            h = g[g.y == c]; ms = h.groupby("seg").l.mean(); ns = h.groupby("seg").size(); k = ns.mean()
            msb = (ns * (ms - h.l.mean()) ** 2).sum() / max(1, len(ms) - 1); msw = ((h.l - h.seg.map(ms)) ** 2).sum() / max(1, len(h) - len(ms))
            iccs.append(max(0.0, (msb - msw) / (msb + (k - 1) * msw)))
        rho = float(np.mean(iccs))
        for s, h in pd.DataFrame(dict(seg=te.seg.values, y=te.truth.values, l=llr_te)).groupby("seg"):
            n = len(h); neff = n / (1 + (n - 1) * rho); y = h.y.iloc[0]
            for k, L in (("independent over time", lp + h.l.sum()), ("design-effect corrected", lp + neff * h.l.mean()), ("segment mean", lp + h.l.mean())):
                seg_out[k].append((y, 1 / (1 + np.exp(-np.clip(L, -50, 50)))))
        say(f"[FUSION] fold {v}: estimated within-segment ICC of frame evidence {rho:.2f}") if v == D.video.unique()[0] else None
    say("[FUSION] frame level (leave-one-video-out):")
    for k, p in preds.items():
        m = metrics(D.truth.values, p); say(f"[FUSION]   {k}: acc {m['acc']:.3f}, Brier {m['brier']:.4f}, NLL {m['nll']:.3f}, ECE {m['ece']:.3f}, overconfidence {m['over']:+.3f}")
    say(f"[FUSION] 25-frame segment level ({len(seg_out['segment mean'])} segments):")
    for k, L in seg_out.items():
        y, p = map(np.array, zip(*L)); m = metrics(y, p)
        say(f"[FUSION]   {k}: acc {m['acc']:.3f}, Brier {m['brier']:.4f}, NLL {m['nll']:.3f}, ECE {m['ece']:.3f}, overconfidence {m['over']:+.3f}")
stage("ANALYZE", analyze)
say("[END] avid.py finished")
