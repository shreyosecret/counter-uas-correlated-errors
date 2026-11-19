"""
Second dataset: Anti-UAV RGB-T (Jiang et al. 2021), public copy on Hugging Face
(datasets/CornBac0n/Anti-UAV-RGBT). Drones only, mostly small and distant,
with SIMULTANEOUS infrared and visible video. Two questions:
  1. Does within-clip error clustering (tau) replicate on small, distant targets?
  2. Do the two sensors fail on the same targets at the same moment?
     (the cross-sensor independence the audit needs; untestable on Halmstad)
Views: zero-shot CLIP, and a CLIP linear probe trained on the Halmstad crops
(out-of-domain transfer), if crops_V/ and crops_IR/ exist in the working dir.
Usage: python antiuav_test.py [n_sequences=60]
"""
import os, sys, json, math, warnings
import numpy as np, pandas as pd, cv2
warnings.filterwarnings("ignore")
from huggingface_hub import list_repo_files, hf_hub_download
REPO = "CornBac0n/Anti-UAV-RGBT"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 60
STEP, CONTEXT, MINC = 50, 3.0, 48            # every 50 frames (2 s at 25 fps)

files = list_repo_files(REPO, repo_type="dataset")
seqs = {}
for f in files:
    parts = f.split("/")
    if len(parts) >= 3 and parts[-1] in ("infrared.mp4", "visible.mp4", "infrared.json", "visible.json"):
        seqs.setdefault("/".join(parts[:-1]), set()).add(parts[-1])
full = sorted(s for s, v in seqs.items() if len(v) == 4)
test = [s for s in full if s.startswith("test/")] or full
print(f"{len(full)} complete sequences; using {min(N, len(test))} from {'test' if test != full else 'all'}", flush=True)

os.makedirs("au_crops", exist_ok=True)
rows = []
for s in test[:N]:
    loc = {k: hf_hub_download(REPO, f"{s}/{k}", repo_type="dataset") for k in ("infrared.mp4", "visible.mp4", "infrared.json", "visible.json")}
    ann = {m: json.load(open(loc[f"{m}.json"])) for m in ("infrared", "visible")}
    caps = {m: cv2.VideoCapture(loc[f"{m}.mp4"]) for m in ("infrared", "visible")}
    n = min(len(ann["infrared"]["exist"]), len(ann["visible"]["exist"]))
    want = [f for f in range(0, n, STEP)
            if all(ann[m]["exist"][f] and len(ann[m]["gt_rect"][f]) == 4 and ann[m]["gt_rect"][f][2] * ann[m]["gt_rect"][f][3] > 0
                   for m in ("infrared", "visible"))]
    for m, cap in caps.items():
        f = 0; wset = set(want)
        while wset:
            ok, img = cap.read()
            if not ok:
                break
            if f in wset:
                x, y, w, h = ann[m]["gt_rect"][f]; H, W = img.shape[:2]
                sz = max(MINC, CONTEXT * max(w, h)); cx, cy = x + w / 2, y + h / 2
                x0, y0 = int(max(0, cx - sz / 2)), int(max(0, cy - sz / 2)); x1, y1 = int(min(W, cx + sz / 2)), int(min(H, cy + sz / 2))
                if x1 - x0 > 4 and y1 - y0 > 4:
                    name = f"{s.replace('/', '_')}__{m}__{f}.jpg"
                    cv2.imwrite(f"au_crops/{name}", cv2.resize(img[y0:y1, x0:x1], (224, 224)))
                    rows.append(dict(file=name, track=s, frame=f, sensor=m, size=math.sqrt(w * h), dist="NA", truth=1))
                wset.discard(f)
            f += 1
        cap.release()
    print("seq", s, len(want), "frames", flush=True)
M = pd.DataFrame(rows)
print(M.groupby("sensor")["size"].describe().round(1).to_string(), flush=True)

# ---------------- CLIP zero-shot + Halmstad-trained probe
import torch, open_clip
from PIL import Image
from sklearn.linear_model import LogisticRegression
dev = "cuda" if torch.cuda.is_available() else "cpu"
cm, _, prep = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k", device=dev)
tok = open_clip.get_tokenizer("ViT-B-32")
with torch.no_grad():
    T = cm.encode_text(tok(["a photo of a drone in the sky", "a photo of a bird in the sky",
                            "a photo of an empty sky", "a photo of an airplane"]).to(dev))
    T = T / T.norm(dim=-1, keepdim=True)


def embed(paths):
    out = []
    with torch.no_grad():
        for i in range(0, len(paths), 64):
            b = torch.stack([prep(Image.open(p).convert("RGB")) for p in paths[i:i + 64]]).to(dev)
            e = cm.encode_image(b); out.append((e / e.norm(dim=-1, keepdim=True)).cpu())
    return torch.cat(out)


E = embed([f"au_crops/{f}" for f in M.file])
M["clip"] = ((100 * E.to(dev) @ T.T).softmax(-1).argmax(-1) == 0).int().cpu().numpy()
for m, hs in (("visible", "V"), ("infrared", "IR")):
    if os.path.exists(f"crops_{hs}/meta.json"):
        hm = json.load(open(f"crops_{hs}/meta.json"))
        HX = embed([f"crops_{hs}/" + r["file"] for r in hm]).numpy(); Hy = np.array([int(r["truth"] == "drone") for r in hm])
        clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000).fit(HX, Hy)
        idx = (M.sensor == m).values
        M.loc[idx, "probe"] = clf.predict(E[idx].numpy())
        print(f"probe trained on Halmstad {hs} ({len(hm)} crops)", flush=True)
M.to_csv("antiuav_votes.csv", index=False)

# ---------------- tau, cross-sensor dependence
from tau_ml import fit, design, profile_ci
from scipy.stats import chi2
from statsmodels.stats.contingency_tables import StratifiedTable
lines = []
def say(x):
    print(x, flush=True); lines.append(x)
say(f"\n=== Anti-UAV RGB-T: {M.track.nunique()} sequences, {len(M) // 2} paired frames ===")
for model in [c for c in ("clip", "probe") if c in M and M[c].notna().all()]:
    for m in ("visible", "infrared"):
        d = M[M.sensor == m].copy(); d["err"] = (d[model] != 1).astype(int)
        t = d.groupby("track").size(); d = d[d.track.isin(t[t >= 3].index)]
        if d.err.nunique() < 2:
            say(f"{model:5s} {m:8s} drone recall {1 - d.err.mean():.3f}  tau not estimable"); continue
        X = design(d); g = pd.factorize(d.track)[0]
        tau, b, nll, lr = fit(X, d.err.values, g); lo, hi = profile_ci(X, d.err.values, g, tau, b, nll)
        say(f"{model:5s} {m:8s} drone recall {1 - d.err.mean():.3f} | tau {tau:.2f} [{lo:.2f}, {hi:.2f}] p={0.5 * chi2.sf(lr, 1):.2g} | {d.track.nunique()} seqs, {len(d)} crops")
    P = M.pivot_table(index=["track", "frame"], columns="sensor", values=model).dropna()
    eI, eV = (P["infrared"] != 1).astype(int), (P["visible"] != 1).astype(int)
    tab = pd.crosstab(eI, eV)
    r = pd.DataFrame(dict(eI=eI, eV=eV)).groupby(level=0).mean()
    tables = []
    for _, g in pd.DataFrame(dict(eI=eI, eV=eV)).groupby(level=0):
        tb = np.array([[((g.eI == 1) & (g.eV == 1)).sum(), ((g.eI == 1) & (g.eV == 0)).sum()],
                       [((g.eI == 0) & (g.eV == 1)).sum(), ((g.eI == 0) & (g.eV == 0)).sum()]], float)
        if tb.sum() >= 2:
            tables.append(tb + 0.5)
    st = StratifiedTable(tables)
    say(f"{model:5s} cross-sensor: error rates IR {eI.mean():.3f}, visible {eV.mean():.3f}; both wrong {((eI == 1) & (eV == 1)).mean():.3f} "
        f"vs {eI.mean() * eV.mean():.3f} if independent")
    say(f"{model:5s} cross-sensor: per-sequence error-rate correlation {r.eI.corr(r.eV):.2f}; "
        f"within-sequence Mantel-Haenszel OR {st.oddsratio_pooled:.2f} (p={st.test_null_odds().pvalue:.2g})")
open("antiuav_results.txt", "w").write("\n".join(lines))
