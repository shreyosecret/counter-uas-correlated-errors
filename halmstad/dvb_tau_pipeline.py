"""
Real-data test of the project's key assumption: do a VLM's errors cluster
within the same drone track (the "hard target" correlation, tau)?

Phase 1 needs NO bird labels: it uses only the public Drone-vs-Bird drone
boxes plus the videos (free after the data-use agreement). A VLM error here
means calling a real drone "not a drone".

Steps
  1. extract : sample crops from each track, spaced >= --spacing seconds
               (matches the 2 s re-query gap in the simulator)
  2. query   : ask a VLM about each crop  (backends: clip, qwen, dummy)
  3. analyze : random-intercept logistic regression
                   error ~ log(box size) + (1 | track)
               tau = SD of the track intercept; ICC and design effect follow.
               A permutation null (shuffle tracks within size bins) shows what
               tau looks like when there is no clustering.

Usage (Colab or Kaggle free GPU):
  python dvb_tau_pipeline.py extract --videos VIDEOS/ --ann challenge/annotations/ --out crops/
  python dvb_tau_pipeline.py query   --crops crops/ --backend clip --out votes_clip.csv
  python dvb_tau_pipeline.py query   --crops crops/ --backend qwen --out votes_qwen.csv
  python dvb_tau_pipeline.py analyze --votes votes_clip.csv votes_qwen.csv
  python dvb_tau_pipeline.py selftest            # end-to-end check on synthetic data

Same model in R (lme4), if you prefer:
  m <- glmer(err ~ log(size) + (1 | track), data = d, family = binomial)
  VarCorr(m)   # SD of track intercept = tau
"""
import argparse, csv, glob, json, math, os, sys
import numpy as np


# ------------------------------------------------------------------ extract
def parse_ann(path):
    """Drone-vs-Bird format: frame n [x y w h cls]*n  ->  {frame: [(x,y,w,h)]}"""
    out = {}
    for line in open(path):
        t = line.split()
        if len(t) < 2:
            continue
        f, n = int(t[0]), int(t[1])
        out[f] = [tuple(float(v) for v in t[2 + 5 * i: 6 + 5 * i]) for i in range(n)]
    return out


def link_tracks(ann, max_jump=80.0, max_gap=15):
    """Greedy nearest-centre linking of boxes into tracks within one video."""
    tracks, active = [], []          # active: (track_id, last_frame, cx, cy)
    for f in sorted(ann):
        used = set(); new_active = []
        for (x, y, w, h) in ann[f]:
            cx, cy = x + w / 2, y + h / 2
            best, bd = None, max_jump
            for j, (tid, lf, ax, ay) in enumerate(active):
                if j in used or f - lf > max_gap:
                    continue
                d = math.hypot(cx - ax, cy - ay)
                if d < bd:
                    best, bd = j, d
            if best is None:
                tid = len(tracks); tracks.append([])
            else:
                tid = active[best][0]; used.add(best)
            tracks[tid].append((f, x, y, w, h))
            new_active.append((tid, f, cx, cy))
        keep = [a for j, a in enumerate(active) if j not in used and f - a[1] <= max_gap]
        active = keep + new_active
    return tracks


def find_video(videos, stem):
    for ext in (".mp4", ".avi", ".mov", ".MP4", ".MOV", ".mkv"):
        p = os.path.join(videos, stem + ext)
        if os.path.exists(p):
            return p
    return None


def extract(a):
    import cv2
    os.makedirs(a.out, exist_ok=True)
    meta = []
    for ap in sorted(glob.glob(os.path.join(a.ann, "*.txt"))):
        stem = os.path.splitext(os.path.basename(ap))[0]
        vp = find_video(a.videos, stem)
        if vp is None:
            print("no video for", stem); continue
        cap = cv2.VideoCapture(vp)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        step = max(1, int(round(a.spacing * fps)))
        want = {}
        for ti, tr in enumerate(link_tracks(parse_ann(ap))):
            if len(tr) < a.min_track:
                continue
            last = -10 ** 9
            for (f, x, y, w, h) in tr:
                if f - last >= step:
                    want.setdefault(f, []).append((ti, x, y, w, h)); last = f
        f = 0
        while want:
            ok, img = cap.read()
            if not ok:
                break
            if f in want:
                H, W = img.shape[:2]
                for (ti, x, y, w, h) in want.pop(f):
                    s = max(a.min_crop, a.context * max(w, h))
                    cx, cy = x + w / 2, y + h / 2
                    x0, y0 = int(max(0, cx - s / 2)), int(max(0, cy - s / 2))
                    x1, y1 = int(min(W, cx + s / 2)), int(min(H, cy + s / 2))
                    crop = cv2.resize(img[y0:y1, x0:x1], (224, 224))
                    name = f"{stem}__t{ti}__f{f}.jpg"
                    cv2.imwrite(os.path.join(a.out, name), crop)
                    meta.append(dict(file=name, video=stem, track=f"{stem}__t{ti}", frame=f,
                                     size=float(np.sqrt(w * h)), truth="drone"))
            f += 1
        cap.release()
    with open(os.path.join(a.out, "meta.json"), "w") as fh:
        json.dump(meta, fh)
    print(f"wrote {len(meta)} crops from {len(set(m['track'] for m in meta))} tracks")


# ------------------------------------------------------------------ query
class ClipBackend:
    """Zero-shot CLIP: cheap, runs on free GPU or even CPU."""
    def __init__(self):
        import torch, open_clip
        self.torch = torch
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _, self.prep = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k", device=self.dev)
        tok = open_clip.get_tokenizer("ViT-B-32")
        prompts = ["a photo of a drone in the sky", "a photo of a bird in the sky",
                   "a photo of an empty sky", "a photo of an airplane"]
        with torch.no_grad():
            t = self.model.encode_text(tok(prompts).to(self.dev))
            self.text = t / t.norm(dim=-1, keepdim=True)

    def __call__(self, pil, sensor=None):
        with self.torch.no_grad():
            im = self.model.encode_image(self.prep(pil).unsqueeze(0).to(self.dev))
            im = im / im.norm(dim=-1, keepdim=True)
            p = (100 * im @ self.text.T).softmax(-1)[0].cpu().numpy()
        return int(p.argmax() == 0), float(p[0])


class QwenBackend:
    """Open-weights VLM. 3B fits a free T4; swap in a newer/larger model if available."""
    def __init__(self, model_id="Qwen/Qwen2.5-VL-3B-Instruct"):
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText
        self.torch = torch
        self.proc = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map="auto")
        self.q = ("This is a crop from a sky camera. Is the small object a drone, a bird, "
                  "or something else? Answer with one word: drone, bird, or other.")

    def __call__(self, pil, sensor=None):
        q = self.q if sensor != "IR" else self.q.replace("a sky camera", "a thermal infrared sky camera")
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]}]
        text = self.proc.apply_chat_template(msgs, add_generation_prompt=True)
        inp = self.proc(text=[text], images=[pil], return_tensors="pt").to(self.model.device)
        with self.torch.no_grad():
            out = self.model.generate(**inp, max_new_tokens=4, do_sample=False)
        ans = self.proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        ans = ans.strip().lower()
        return int(ans.startswith("drone")), float(ans.startswith("drone"))


class DummyBackend:
    """Simulated VLM with a known tau, so the analysis can be checked."""
    def __init__(self, tau=1.5, seed=0):
        self.tau, self.rng, self.u = tau, np.random.default_rng(seed), {}

    def vote(self, track, size, truth=1):
        if track not in self.u:
            self.u[track] = self.rng.normal()
        p_err = 1 / (1 + math.exp(-(1.0 - 1.2 * math.log(size / 10) + self.tau * self.u[track])))
        v = truth if self.rng.random() > p_err else 1 - truth
        return v, float(v)


def query(a):
    meta = json.load(open(os.path.join(a.crops, "meta.json")))
    done = set()
    if os.path.exists(a.out):                       # resume after a Colab timeout
        done = {r["file"] for r in csv.DictReader(open(a.out))}
    be = {"clip": ClipBackend, "qwen": QwenBackend, "dummy": DummyBackend}[a.backend]()
    new = not os.path.exists(a.out)
    with open(a.out, "a", newline="") as fh:
        w = csv.DictWriter(fh, ["file", "video", "track", "frame", "size", "vote", "score", "backend",
                                "truth", "sensor", "dist"])
        if new:
            w.writeheader()
        for i, m in enumerate(meta):
            if m["file"] in done:
                continue
            truth = int(m.get("truth", "drone") == "drone")
            if a.backend == "dummy":
                v, s = be.vote(m["track"], m["size"], truth)
            else:
                from PIL import Image
                v, s = be(Image.open(os.path.join(a.crops, m["file"])).convert("RGB"), m.get("sensor"))
            w.writerow(dict(file=m["file"], video=m["video"], track=m["track"], frame=m["frame"],
                            size=m["size"], vote=v, score=s, backend=a.backend, truth=truth,
                            sensor=m.get("sensor", ""), dist=m.get("dist", "")))
            if i % 200 == 0:
                fh.flush(); print(f"{i}/{len(meta)}")


# ------------------------------------------------------------------ analyze
def fit_tau(err, logsize, groups, dist=None):
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
    import pandas as pd
    df = pd.DataFrame(dict(err=err, ls=logsize, g=groups))
    form = "err ~ ls"
    if dist is not None and pd.Series(dist).nunique() > 1:
        df["dist"] = dist; form = "err ~ ls + C(dist)"
    m = BinomialBayesMixedGLM.from_formula(form, {"g": "0 + C(g)"}, df).fit_vb()
    return float(np.exp(m.vcp_mean[0])), m


def add_err(d):
    """Error = vote disagrees with truth. Old drone-only files: every crop is a drone."""
    truth = d["truth"].astype(int) if "truth" in d and d["truth"].notna().all() else 1
    d["err"] = (d["vote"].astype(int) != truth).astype(int)
    d["cls"] = np.where(truth == 1, "drones", "birds") if not np.isscalar(truth) else "drones"
    return d


def analyze(a):
    import pandas as pd
    for path in a.votes:
        d = add_err(pd.read_csv(path))
        d["ls"] = np.log(d["size"])
        tsz = d.groupby("track").size()
        d = d[d["track"].isin(tsz[tsz >= 3].index)]       # need repeats to see clustering
        print(f"\n=== {path}: {len(d)} crops, {d.track.nunique()} tracks ===")
        bins = pd.cut(d["size"], [0, 10, 16, 32, 64, 1e9], labels=["<10", "10-16", "16-32", "32-64", "64+"])
        print("error rate by box size (px):")
        print(d.groupby(bins, observed=True)["err"].agg(["mean", "size"]).round(3).to_string())
        dist = d["dist"].astype(str).values if "dist" in d else None
        for cname in sorted(d["cls"].unique()) if d["cls"].nunique() > 1 else []:
            S = d[d["cls"] == cname]
            t_c, _ = fit_tau(S["err"].values, S["ls"].values, S["track"].values,
                             None if dist is None else S["dist"].astype(str).values)
            print(f"  tau [{cname}] = {t_c:.2f}   ({S.track.nunique()} tracks, error rate {S.err.mean():.3f})")
        tau, m = fit_tau(d["err"].values, d["ls"].values, d["track"].values, dist)
        icc = tau ** 2 / (tau ** 2 + math.pi ** 2 / 3)
        print(f"tau (track-level SD, logit scale) = {tau:.2f}   latent ICC = {icc:.2f}")
        for k in (2, 3, 5):
            print(f"  {k} repeat queries are worth about {k / (1 + (k - 1) * icc):.2f} independent ones")
        rng = np.random.default_rng(0); null = []
        for _ in range(a.perm):
            g = d["track"].values.copy()
            for b in bins.cat.categories:
                idx = np.where(bins.values == b)[0]
                g[idx] = rng.permutation(g[idx])
            null.append(fit_tau(d["err"].values, d["ls"].values, g)[0])
        if null:
            print(f"permutation null tau: median {np.median(null):.2f}, 95th pct {np.quantile(null, .95):.2f}"
                  f"  -> observed tau {'EXCEEDS' if tau > np.quantile(null, .95) else 'does not exceed'} null")
    if len(a.votes) == 2:
        x, y = [add_err(pd.read_csv(p)) for p in a.votes]
        j = x.merge(y, on="file", suffixes=("_a", "_b"))
        r = j.rename(columns={"err_a": "ea", "err_b": "eb"}).groupby("track_a")[["ea", "eb"]].mean()
        r = r[j.groupby("track_a").size() >= 3]
        print(f"\ncross-model: correlation of per-track error rates = {r.ea.corr(r.eb):.2f} "
              f"(near 0 = independent failures, near 1 = they fail on the same tracks)")


# ------------------------------------------------------------------ selftest
def selftest(a):
    """Build a tiny synthetic video + annotation, run all three steps with the
    dummy backend at known tau, and check the estimate."""
    import cv2, tempfile
    root = tempfile.mkdtemp(); vids, ann = os.path.join(root, "v"), os.path.join(root, "a")
    os.makedirs(vids); os.makedirs(ann)
    rng = np.random.default_rng(1)
    for k in range(60):
        name = f"vid{k:02d}"; fps, n = 10, 200
        vw = cv2.VideoWriter(os.path.join(vids, name + ".avi"), cv2.VideoWriter_fourcc(*"MJPG"), fps, (320, 240))
        x, y, s = rng.uniform(40, 280), rng.uniform(40, 200), rng.uniform(6, 60)
        vx, vy = rng.normal(0, 1.5, 2)
        with open(os.path.join(ann, name + ".txt"), "w") as fh:
            for f in range(n):
                img = np.full((240, 320, 3), 200, np.uint8)
                x = float(np.clip(x + vx, 10, 300)); y = float(np.clip(y + vy, 10, 220))
                cv2.rectangle(img, (int(x), int(y)), (int(x + s / 3), int(y + s / 3)), (40, 40, 40), -1)
                vw.write(img)
                fh.write(f"{f} 1 {x:.1f} {y:.1f} {s:.1f} {s:.1f} drone\n")
        vw.release()
    crops = os.path.join(root, "c")
    ns = argparse.Namespace
    extract(ns(videos=vids, ann=ann, out=crops, spacing=2.0, min_track=10, context=3.0, min_crop=64))
    for tau in (0.0, 1.5):
        out = os.path.join(root, f"votes_{tau}.csv")
        DummyBackend.__init__.__defaults__ = (tau, 0)
        query(ns(crops=crops, backend="dummy", out=out))
        print(f"\n[selftest] true tau = {tau}")
        analyze(ns(votes=[out], perm=a.perm))


if __name__ == "__main__":
    p = argparse.ArgumentParser(); sp = p.add_subparsers(dest="cmd", required=True)
    e = sp.add_parser("extract"); e.add_argument("--videos", required=True); e.add_argument("--ann", required=True)
    e.add_argument("--out", required=True); e.add_argument("--spacing", type=float, default=2.0)
    e.add_argument("--min_track", type=int, default=10); e.add_argument("--context", type=float, default=3.0)
    e.add_argument("--min_crop", type=int, default=64)
    q = sp.add_parser("query"); q.add_argument("--crops", required=True)
    q.add_argument("--backend", choices=["clip", "qwen", "dummy"], required=True); q.add_argument("--out", required=True)
    z = sp.add_parser("analyze"); z.add_argument("--votes", nargs="+", required=True); z.add_argument("--perm", type=int, default=20)
    s = sp.add_parser("selftest"); s.add_argument("--perm", type=int, default=5)
    a = p.parse_args()
    {"extract": extract, "query": query, "analyze": analyze, "selftest": selftest}[a.cmd](a)
