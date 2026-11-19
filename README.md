# Correlated errors in counter-UAS perception

Code for: S. Ghosh, *Correlated Verifier Errors in Counter-UAS Perception: Clustering, Cross-Sensor Dependence, and Label-Free Auditing* (submitted to *Data-Centric Engineering*; preprint on engrXiv).

All analyses use public data only. No data are redistributed here except small per-crop vote tables derived from the Halmstad dataset.

## Datasets (download separately)
| Dataset | Source | Used for |
|---|---|---|
| Halmstad multi-sensor drone dataset | Svanström et al., *Data in Brief* 39:107521 (2021) | Zero-shot CLIP, Qwen2.5-VL, linear probe, HOG; query-policy replay; scene split |
| Anti-UAV RGB-T | Jiang et al., *IEEE Trans. Multimedia*; public copy `CornBac0n/Anti-UAV-RGBT` on Hugging Face | Trained detectors, session analyses, cross-sensor dependence |
| AVID | `Dharini24/AVID` on Hugging Face (CC BY-NC 4.0), a paired subset of the Halmstad recordings | Real negatives (birds), audit with specificity, fusion, lag decay |

## Layout and paper mapping
| Folder / script | What it does | Paper section |
|---|---|---|
| `halmstad/` | Crop extraction, zero-shot and probe votes, exact-likelihood tau estimation (`tau_ml.py`), audit on Halmstad (`audit_real.py`) | Results: clustering, zero-shot VLMs, adaptation, Halmstad audit |
| `antiuav/attack.py` | YOLO11n training and evaluation on Anti-UAV RGB-T; exact random-effects models (shared by other scripts) | Trained detector; sequence vs session |
| `antiuav/extra.py` | Pooled odds ratio, 1280 px check, real-data audit votes, extra seeds, Halmstad scene groups | Cross-sensor dependence; audit on real votes |
| `antiuav/extra2.py` | YOLO11s and RT-DETR-L | Other architectures |
| `antiuav/sessions2.py` | Profile-likelihood intervals; two-fold session cross-fitting over all 37 sessions | Session cross-fitting; sequence vs session |
| `avid/avid.py`, `avid/fix12.py` | AVID scoring, audit with negatives, fusion rules, lag-decay analysis | Real negatives; combining correlated evidence |
| `analysis/dce_checks.py` | Fold-adjusted session model, session-cluster bootstrap for the odds ratio, AVID complete-segment fusion and video-level bootstrap | Robustness checks requested in review |
| `analysis/policy_replay.py` | Sequential query policies replayed on real Halmstad votes | Correlation-aware querying on real votes |
| `analysis/scene_split.py` | Nested scene/clip models for learned verifiers on Halmstad | Hard targets and hard scenes |
| `analysis/audit_sim.py` | Calibrated audit simulation | Appendix |
| `figures/` | All figures (`make_figures.py` for data figures, `fig_overview.py` for Figure 1) | Figures |
| `results/` | Halmstad per-crop votes and scene groups (inputs to `analysis/`) | |

## Running
The GPU stages were run in Google Colab on a T4. Scripts in `antiuav/` and `avid/` read and write a working folder set by the `WORK` variable at the top of each file (`/content/drive/MyDrive/counter_uas_attack` in Colab); change it to run elsewhere. Order: `attack.py` → `extra.py` → `extra2.py` → `sessions2.py` → `avid.py` → `fix12.py` → `analysis/dce_checks.py`. Every stage is resumable: rerunning a script skips completed stages.

Fixed seeds: session split seed 7; bootstrap seed 2026; simulation seeds are set at the top of each script. Detector training uses Ultralytics defaults (deterministic mode) with the epochs and image sizes stated in the paper.

The scripts in `analysis/` run on a laptop: `cd analysis && python policy_replay.py`.

## Requirements
Python 3.10+, see `requirements.txt`.

## Licence
MIT (code). Datasets keep their own licences; AVID is non-commercial (CC BY-NC 4.0).
