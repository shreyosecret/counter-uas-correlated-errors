"""
Reader for the open (CC0) Halmstad multi-sensor drone dataset:
  github.com/DroneDetectionThesis/Drone-detection-dataset
Handles both sensors: visible (Data/Video_V, prefix V_) and thermal infrared
(Data/Video_IR, prefix IR_). One target class per clip. The description sheet
gives a distance bin (CLOSE / MEDIUM / DISTANT) per clip, used as a difficulty
covariate.

extract_crops() writes 224x224 JPEG crops plus meta.json in the format that
dvb_tau_pipeline.py `query` and `analyze` read, so the same CLIP / VLM backends
work on this dataset unchanged.
"""
import glob, json, math, os
import numpy as np
import cv2

SENSOR_DIR = {"V": "Video_V", "IR": "Video_IR"}


def load_description(root):
    import pandas as pd
    d = pd.read_excel(os.path.join(root, "Video_dataset_description.xlsx"))
    d.columns = ["sensor", "cls", "num", "dist", "bin", "type"]
    out = {}
    for _, r in d.iterrows():
        out[f"{str(r.sensor).strip()}_{str(r.cls).strip()}_{int(r.num):03d}"] = \
            (str(r.dist).strip(), None if str(r.type) == "nan" else str(r.type).strip())
    return out


def iter_clips(root, sensor, classes=("DRONE", "BIRD")):
    from mcos_decoder import load_groundtruth
    desc = load_description(root)
    for cls in classes:
        pat = os.path.join(root, SENSOR_DIR[sensor], f"{sensor}_{cls}_*_LABELS.mat")
        for lab in sorted(glob.glob(pat)):
            name = os.path.basename(lab).replace("_LABELS.mat", "")
            try:
                boxes = load_groundtruth(lab)
            except Exception:
                print("skipping unreadable labels:", name); continue
            dist, dtype = desc.get(name, ("UNKNOWN", None))
            yield dict(name=name, sensor=sensor, y=int(cls == "DRONE"), dist=dist,
                       dtype=dtype, boxes=boxes, video=lab.replace("_LABELS.mat", ".mp4"))


def motion_features(boxes, fps):
    B = np.array([b for b in boxes if b is not None], float)
    if len(B) < int(fps):
        return None
    c = B[:, :2] + B[:, 2:] / 2
    size = np.sqrt(B[:, 2] * B[:, 3])
    step = np.linalg.norm(np.diff(c, axis=0), axis=1)
    win = int(fps)
    st = [np.linalg.norm(c[i + win] - c[i]) / max(step[i:i + win].sum(), 1e-6)
          for i in range(0, len(c) - win, max(1, win // 2)) if step[i:i + win].sum() > 1]
    dla = np.diff(np.log(B[:, 2] * B[:, 3] + 1e-6))
    dar = np.diff(np.log(B[:, 2] / (B[:, 3] + 1e-6) + 1e-6))
    return dict(size=float(np.median(size)), speed=float(np.median(step / size[1:])),
                straight=float(np.median(st)) if st else np.nan,
                area_flicker=float(np.std(dla)), aspect_flicker=float(np.std(dar)),
                size_var=float(np.std(np.log(size))))


def extract_crops(root, sensor, out_dir=None, spacing=2.0, context=3.0, min_crop=48,
                  hog_feats=False):
    """Crops spaced >= `spacing` seconds per clip. Returns (tracks, crops).
    If out_dir is given, writes JPEGs + meta.json for dvb_tau_pipeline.py."""
    hog = cv2.HOGDescriptor((64, 64), (16, 16), (8, 8), (8, 8), 9) if hog_feats else None
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    tracks, crops = [], []
    for clip in iter_clips(root, sensor):
        cap = cv2.VideoCapture(clip["video"]); fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        m = motion_features(clip["boxes"], fps)
        if m is None:
            cap.release(); continue
        tracks.append(dict(track=clip["name"], y=clip["y"], dist=clip["dist"], **m))
        step = int(round(spacing * fps)); last, f = -10 ** 9, 0
        while True:
            ok, img = cap.read()
            if not ok:
                break
            b = clip["boxes"][f] if f < len(clip["boxes"]) else None
            if b is not None and f - last >= step:
                x, y, w, h = b; H, W = img.shape[:2]
                s = max(min_crop, context * max(w, h)); cx, cy = x + w / 2, y + h / 2
                x0, y0 = int(max(0, cx - s / 2)), int(max(0, cy - s / 2))
                x1, y1 = int(min(W, cx + s / 2)), int(min(H, cy + s / 2))
                if x1 - x0 > 4 and y1 - y0 > 4:
                    patch = img[y0:y1, x0:x1]
                    rec = dict(track=clip["name"], video=clip["name"], frame=f, y=clip["y"],
                               truth="drone" if clip["y"] else "bird", size=math.sqrt(w * h),
                               sensor=sensor, dist=clip["dist"])
                    if out_dir:
                        rec["file"] = f"{clip['name']}__f{f}.jpg"
                        cv2.imwrite(os.path.join(out_dir, rec["file"]), cv2.resize(patch, (224, 224)))
                    if hog is not None:
                        g = cv2.cvtColor(cv2.resize(patch, (64, 64)), cv2.COLOR_BGR2GRAY)
                        rec["feat"] = hog.compute(g).ravel()
                    crops.append(rec); last = f
            f += 1
        cap.release()
    if out_dir:
        with open(os.path.join(out_dir, "meta.json"), "w") as fh:
            json.dump([{k: v for k, v in c.items() if k != "feat"} for c in crops], fh)
    return tracks, crops


def fit_tau_cov(df, formula="err ~ ls"):
    """Random-intercept logistic regression; returns the track-level SD (tau)."""
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
    m = BinomialBayesMixedGLM.from_formula(formula, {"g": "0 + C(track)"}, df).fit_vb()
    return float(np.exp(m.vcp_mean[0]))


def tau_with_null(df, formula="err ~ ls", perms=20, seed=0):
    """tau plus a permutation null that shuffles tracks within size x distance cells."""
    tau = fit_tau_cov(df, formula)
    rng = np.random.default_rng(seed)
    cell = (np.digitize(df["size"], [10, 16, 32, 64]).astype(str) + "_" + df["dist"].astype(str)).values
    null = []
    for _ in range(perms):
        g = df["track"].values.copy()
        for c in np.unique(cell):
            idx = np.where(cell == c)[0]; g[idx] = rng.permutation(g[idx])
        null.append(fit_tau_cov(df.assign(track=g), formula))
    return tau, float(np.quantile(null, 0.95))
