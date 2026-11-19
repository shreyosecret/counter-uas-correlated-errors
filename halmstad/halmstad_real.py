"""
Real-data test on the open (CC0) Halmstad drone dataset, CPU only, no model
downloads. Run per sensor:  python halmstad_real.py Data V   |   python halmstad_real.py Data IR
  A. MOTION view: track-level motion features, drone vs bird.
  B. APPEARANCE view: HOG + logistic regression, video-grouped CV. Out-of-fold
     crop errors (>= 2 s apart) -> tau with size + distance covariates and a
     permutation null.
  C. CROSS-VIEW: do appearance and motion fail on the same clips?
Writes hog_errors_<sensor>.csv for later cross-sensor / cross-model comparison.
"""
import math, os, sys
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from halmstad_lib import extract_crops, tau_with_null

root, sensor = (sys.argv[1], sys.argv[2]) if len(sys.argv) > 2 else ("Data", "V")
tracks, crops = extract_crops(root, sensor, hog_feats=True)
T = pd.DataFrame(tracks).dropna(); C = pd.DataFrame(crops); C = C[C.track.isin(T.track)]
print(f"[{sensor}] clips: {len(T)} ({T.y.sum()} drone, {(T.y == 0).sum()} bird); crops: {len(C)}")
print("clips by distance bin:", T.groupby(["dist", "y"]).size().unstack(fill_value=0)
      .rename(columns={0: "bird", 1: "drone"}).to_dict("index"))

feats = ["speed", "straight", "area_flicker", "aspect_flicker", "size_var"]
print("\nmedian motion features:")
print(T.groupby("y")[feats + ["size"]].median().rename(index={0: "bird", 1: "drone"}).round(3).to_string())
mot = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced"))
pm = cross_val_predict(mot, T[feats].values, T.y.values,
                       cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
T["mot_err"] = ((pm > 0.5).astype(int) != T.y).astype(int)
print(f"\nA. MOTION: accuracy {1 - T.mot_err.mean():.3f} "
      f"(drone {1 - T[T.y == 1].mot_err.mean():.3f}, bird {1 - T[T.y == 0].mot_err.mean():.3f})")

X = np.vstack(C.feat.values)
app = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, class_weight="balanced", max_iter=3000))
pa = cross_val_predict(app, X, C.y.values, cv=GroupKFold(5), groups=C.track.values,
                       method="predict_proba")[:, 1]
C["err"] = ((pa > 0.5).astype(int) != C.y.values).astype(int)
print(f"\nB. APPEARANCE (HOG): crop accuracy {1 - C.err.mean():.3f} "
      f"(drone {1 - C[C.y == 1].err.mean():.3f}, bird {1 - C[C.y == 0].err.mean():.3f})")
print("   error by distance bin:", C.groupby("dist").err.agg(["mean", "size"]).round(3).to_dict("index"))
tsz = C.groupby("track").size(); D = C[C.track.isin(tsz[tsz >= 3].index)].copy()
D["ls"] = np.log(D["size"])
for nm, S in [("all", D), ("drones", D[D.y == 1]), ("birds", D[D.y == 0])]:
    for label, form in [("size", "err ~ ls"), ("size+dist", "err ~ ls + C(dist)")]:
        tau, q95 = tau_with_null(S, form)
        icc = tau ** 2 / (tau ** 2 + math.pi ** 2 / 3)
        print(f"   tau [{nm:6s} | {label:9s}] = {tau:.2f} (null 95th {q95:.2f})  ICC {icc:.2f}  "
              f"3 queries ~ {3 / (1 + 2 * icc):.2f}  [{S.track.nunique()} clips, {len(S)} crops]")

J = T.set_index("track").join(C.groupby("track").err.mean().rename("app_err"), how="inner")
print(f"\nC. CROSS-VIEW: corr(appearance error rate, motion error) = {np.corrcoef(J.app_err, J.mot_err)[0, 1]:.2f}")
C.drop(columns="feat").to_csv(f"hog_errors_{sensor}.csv", index=False)
