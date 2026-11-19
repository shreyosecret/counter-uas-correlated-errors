"""Regenerate all data figures for the paper in one consistent style.
Palette (Okabe-Ito, colorblind safe): infrared = vermillion, visible = blue, highlight = green."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, pandas as pd
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
from matplotlib.lines import Line2D

IR, VIS, HL, INK, MID, LIGHT, BAND = "#D55E00", "#0072B2", "#009E73", "#1a1a1a", "#6b6b6b", "#bdbdbd", "#f3f3f3"
plt.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 9, "axes.titlesize": 9.5,
    "axes.labelsize": 9, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.7, "axes.edgecolor": "#444444",
    "xtick.major.width": 0.7, "ytick.major.width": 0.7, "xtick.color": "#333333", "ytick.color": "#333333",
    "axes.titleweight": "bold", "savefig.bbox": "tight", "savefig.pad_inches": 0.03, "pdf.fonttype": 42})

def panel(ax, letter, x=-0.02, y=1.02):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=11, fontweight="bold", va="bottom", ha="right")

def xgrid(ax):
    ax.grid(axis="x", color="#e6e6e6", lw=0.6, zorder=0); ax.set_axisbelow(True)

def ygrid(ax):
    ax.grid(axis="y", color="#e6e6e6", lw=0.6, zorder=0); ax.set_axisbelow(True)

XFIT_OR = (6.26, 3.46, 10.57)   # updated from dce_checks session-cluster bootstrap
sensor_legend = [Line2D([], [], color=IR, marker="o", ls="-", ms=5, label="Infrared"),
                 Line2D([], [], color=VIS, marker="s", ls="-", ms=4.5, label="Visible")]

# ===================================================================== 1. tau forest plot
groups = [
    ("Halmstad (clip-level)", [("HOG baseline", (1.29, .79, 1.92), (1.40, .95, 1.97)),
                                ("CLIP, zero-shot", (1.45, 1.01, 2.02), (1.01, .76, 1.30)),
                                ("Qwen2.5-VL, yes/no", (.74, .42, 1.07), None),
                                ("CLIP, linear probe", (1.37, .86, 2.07), (2.62, 1.85, 3.82))]),
    ("Anti-UAV RGB-T (sequence-level)", [("CLIP, zero-shot", (1.18, .91, 1.53), (1.30, 1.01, 1.69)),
                                         ("YOLO11n, test A", (1.74, 1.18, 2.79), (1.90, 1.38, 2.79)),
                                         ("YOLO11n, test B", (2.45, 1.56, 3.40), (3.68, 2.98, 4.66)),
                                         ("YOLO11s, test B", (3.03, 2.16, 4.72), (3.09, 1.83, 4.65)),
                                         ("RT-DETR-L, test B", (2.44, 1.27, 3.46), (5.08, 3.17, 6.00)),
                                         ("YOLO11n, all 37 sessions*", (1.94, 1.50, 2.44), (2.62, 2.20, 3.09)),
                                         ("YOLO11n, far frames", (3.70, 2.24, 20.86), (3.19, 1.55, 4.66))]),
    ("AVID, cross-dataset (25-frame segments)", [("YOLO11n (Anti-UAV-trained)", (7.24, 6.17, 8.31), (3.38, 2.84, 4.02))])]
XMAX = 9.0
nrow = sum(len(g[1]) for g in groups) + len(groups)
fig, ax = plt.subplots(figsize=(6.4, 0.27 * nrow + 0.7))
y = nrow - 0.5; ticks, labels = [], []
for gi, (gname, rows) in enumerate(groups):
    ax.text(-0.01, y, gname, transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=8.5, fontweight="bold", color=INK)
    y -= 1
    for ri, (lab, vis, ir) in enumerate(rows):
        if ri % 2 == 0: ax.add_patch(Rectangle((0, y - .5), XMAX, 1, color=BAND, lw=0, zorder=0))
        for est, col, mk, dy in ((vis, VIS, "s", .17), (ir, IR, "o", -.17)):
            if est is None: continue
            t, lo, hi = est
            ax.plot([lo, min(hi, XMAX)], [y + dy] * 2, color=col, lw=1.5, solid_capstyle="round", zorder=3)
            ax.plot(t, y + dy, mk, color=col, ms=4.6 if mk == "o" else 4.2, mec="white", mew=.6, zorder=4)
            if hi > XMAX: ax.annotate("", xy=(XMAX, y + dy), xytext=(XMAX - .35, y + dy), arrowprops=dict(arrowstyle="-|>", color=col, lw=1.2), zorder=4)
        ticks.append(y); labels.append(lab); y -= 1
ax.set_yticks(ticks); ax.set_yticklabels(labels); ax.tick_params(axis="y", length=0, pad=4)
ax.set_xlim(0, XMAX); ax.set_ylim(y + .5, nrow); xgrid(ax)
ax.set_xlabel(r"Drone-miss clustering $\tau$ (random-effect SD, logit scale) with 95% profile CI")
_icc = lambda t: np.asarray(t, float) ** 2 / (np.asarray(t, float) ** 2 + np.pi ** 2 / 3)
_inv = lambda r: np.sqrt(np.clip(np.asarray(r, float), 0, .999) * np.pi ** 2 / 3 / (1 - np.clip(np.asarray(r, float), 0, .999)))
sec = ax.secondary_xaxis("top", functions=(_icc, _inv)); sec.set_ticks([.1, .3, .5, .7, .8, .9, .95])
sec.set_xlabel("Correlation between two looks at the same track (latent ICC)", fontsize=8.5, color="#555555", labelpad=4)
sec.tick_params(colors="#555555", labelsize=8)
fig.text(0.99, 0.005, "*out-of-session predictions from session cross-fitting", ha="right", va="bottom", fontsize=7, color=MID)
ax.legend(handles=sensor_legend, frameon=True, framealpha=1, edgecolor="white", loc="upper right")
for sp in ("left",): ax.spines[sp].set_visible(False)
fig.savefig("fig_tau.pdf"); plt.close(fig)

# ===================================================================== 2. cross-sensor odds ratios (forest-plot table)
rows = [("Anti-UAV RGB-T", [("YOLO11n, test A", 7.97, 4.01, 15.86, "o"), ("YOLO11n, test B", 2.45, 1.08, 5.58, "o"),
                            ("YOLO11n, A + B pooled", 4.98, 2.89, 8.60, "o"), ("YOLO11n, seed 1", 18.64, 4.63, 74.95, "o"),
                            ("YOLO11n, seed 2", 7.24, 1.94, 27.05, "o"), ("YOLO11s", 5.80, 2.33, 14.43, "o"),
                            ("RT-DETR-L", 4.57, 0.49, 42.52, "o"),
                            ("YOLO11n, all 315 sequences*", XFIT_OR[0], XFIT_OR[1], XFIT_OR[2], "P")]),
        ("AVID (cross-dataset)", [("YOLO11n, drone frames", 11.98, 1.59, 90.21, "o")])]
nrow = sum(len(r[1]) for r in rows) + len(rows)
fig = plt.figure(figsize=(6.6, 0.27 * nrow + 0.95)); ax = fig.add_axes([0.30, 0.16, 0.50, 0.80])
y = nrow - .5; ticks, labels = [], []
for gname, rs in rows:
    ax.text(-0.02, y, gname, transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=8.5, fontweight="bold"); y -= 1
    for i_, (lab, o, lo, hi, mk) in enumerate(rs):
        if i_ % 2 == 0: ax.add_patch(Rectangle((0.3, y - .5), 300, 1, color=BAND, lw=0, zorder=0, clip_on=False))
        c = HL if mk == "P" else INK
        if np.isfinite(lo): ax.plot([lo, hi], [y, y], color=c, lw=1.7 if mk == "P" else 1.3, solid_capstyle="round", zorder=3)
        ax.plot(o, y, "D" if mk == "P" else "s", color=c, ms=6.2 if mk == "P" else 4.6, mec="white", mew=.6, zorder=4)
        ci = f"[{lo:.2f}, {hi:.2f}]" if np.isfinite(lo) else "[CI pending]"
        ax.text(1.04, y, f"{o:5.2f}  {ci}", transform=ax.get_yaxis_transform(), va="center", fontsize=8, color=c,
                fontweight="bold" if mk == "P" else "normal", family="STIXGeneral")
        ticks.append(y); labels.append(lab); y -= 1
ax.text(1.04, nrow - .1, "OR  [95% CI]", transform=ax.get_yaxis_transform(), va="top", fontsize=8, fontweight="bold")
ax.axvline(1, color=MID, ls=(0, (3, 2)), lw=.9, zorder=2)
ax.set_xscale("log"); ax.set_xlim(0.3, 120); ax.minorticks_off()
ax.set_xticks([0.5, 1, 2, 5, 10, 20, 50, 100]); ax.set_xticklabels(["0.5", "1", "2", "5", "10", "20", "50", "100"])
ax.set_yticks(ticks); ax.set_yticklabels(labels); ax.tick_params(axis="y", length=0, pad=4); ax.spines["left"].set_visible(False)
ax.set_ylim(y + .5, nrow); xgrid(ax)
ax.text(1.06, nrow - .15, "independence", fontsize=7, color=MID, va="top")
ax.set_xlabel("Odds ratio: infrared miss given a visible miss on the same frame (log scale)")
fig.text(0.30, 0.01, "*out-of-session predictions; interval from a session-cluster bootstrap", fontsize=7, color=MID)
fig.savefig("fig_or.pdf"); plt.close(fig)

# ===================================================================== 3. audit simulation
OR = np.array([1, 2.5, 8])
fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.75), sharey=True)
spec = [(axs[0], "300 tracks per audit", dict(DS=([3.1, 3.4, 5.1], [1.3, 1.6, 3.2], [5.8, 5.7, 6.9]), PAIR=([4.1, 4.1, 3.9], [1.6, 1.5, 1.5], [8.2, 7.7, 7.5]),
                                              LAB=([6.7, 6.6, 6.6], [2.7, 2.8, 2.8], [11.7, 11.8, 11.9]))),
        (axs[1], "5,000 tracks per audit", dict(DS=([.8, 2.0, 5.2], None, None), PAIR=([1.3, 1.2, 1.2], None, None)))]
style = dict(DS=(IR, "o", "-", "Dawid-Skene (assumes independence)"), PAIR=(VIS, "s", "-", "Dependence-aware (joint IR-visible view)"),
             LAB=(MID, "^", "--", "30 labeled tracks only"))
for ax, title, d in spec:
    for k, (m, lo, hi) in d.items():
        c, mk, ls, lab = style[k]
        if lo is not None and k != "LAB": ax.fill_between(OR, lo, hi, color=c, alpha=.07, lw=0)
        ax.plot(OR, m, ls=ls, marker=mk, color=c, lw=1.6, ms=5, mec="white", mew=.6, label=lab)
    ax.set_xscale("log"); ax.minorticks_off(); ax.set_xticks(OR); ax.set_xticklabels(["1", "2.5", "8"])
    ax.set_title(title); ax.set_xlabel("Infrared-visible error odds ratio"); ygrid(ax)
    ax.axvspan(2.45, 7.97, color="#fff4e5", zorder=0); ax.text(4.4, ax.get_ylim()[1] if False else 12.6, "measured range", ha="center", fontsize=7.2, color="#a35a00")
axs[0].set_ylabel("Audit error (mean abs. error, points)\nIR and visible sensitivity, specificity"); axs[0].set_ylim(0, 13.2)
panel(axs[0], "a"); panel(axs[1], "b")
h, l = axs[0].get_legend_handles_labels(); fig.legend(h, l, frameon=False, loc="upper center", ncol=3, bbox_to_anchor=(.5, -.01))
fig.savefig("fig_audit.pdf"); plt.close(fig)

# ===================================================================== 4. query-policy replay (small multiples)
S = pd.read_csv("policy_sweep.csv")
names = {"clip_IR": "CLIP, infrared", "clip_V": "CLIP, visible", "probe_IR": "Probe, infrared", "probe_V": "Probe, visible"}
fig = plt.figure(figsize=(6.6, 3.3)); gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.25], wspace=.42, hspace=.55)
for k, v in enumerate(names):
    ax = fig.add_subplot(gs[k // 2, k % 2]); g = S[S.verifier == v]
    for p, ls, mk, c in (("IND", (0, (3, 2)), "o", LIGHT if False else "#9a9a9a"), ("CORR", "-", "s", IR if "IR" in v else VIS)):
        h = g[g.policy == p].sort_values("q")
        ax.plot(h.q, h.acc, ls=ls, marker=mk, ms=3.4, color=c, lw=1.4, mec="white", mew=.4)
    ax.set_title(names[v], fontsize=8.3, loc="left"); ygrid(ax); ax.tick_params(labelsize=7.5)
    lo, hi = g.acc.min(), g.acc.max(); pad = max(.01, (hi - lo) * .25); ax.set_ylim(lo - pad, hi + pad)
    if k % 2 == 0: ax.set_ylabel("Accuracy", fontsize=8)
    if k // 2 == 1: ax.set_xlabel("Queries per clip", fontsize=8)
    if k == 0: panel(ax, "a", x=-.28, y=1.08)
fig.legend(handles=[Line2D([], [], color="#9a9a9a", ls=(0, (3, 2)), marker="o", ms=3.5, label="independence policy"),
                    Line2D([], [], color=INK, marker="s", ms=3.5, label="correlation-aware policy")],
           frameon=False, fontsize=7.4, loc="lower left", bbox_to_anchor=(0.06, -0.06), ncol=2)
ax = fig.add_subplot(gs[:, 2]); t = S[S.thr == .95]; vs = list(names); x = np.arange(len(vs)); w = .38
ind = [t[(t.verifier == v) & (t.policy == "IND")].over.iloc[0] * 100 for v in vs]
cor = [t[(t.verifier == v) & (t.policy == "CORR")].over.iloc[0] * 100 for v in vs]
ax.bar(x - w / 2, ind, w, color=LIGHT, label="independence", zorder=3); ax.bar(x + w / 2, cor, w, color="#4d4d4d", label="correlation-aware", zorder=3)
for xi, a_, b_ in zip(x, ind, cor):
    ax.text(xi - w / 2, a_ + .3, f"{a_:.0f}", ha="center", fontsize=7, color=MID); ax.text(xi + w / 2, max(b_, 0) + .3, f"{b_:.0f}", ha="center", fontsize=7)
ax.axhline(0, color=INK, lw=.7); ax.set_xticks(x); ax.set_xticklabels([names[v].replace(", ", "\n") for v in vs], fontsize=7)
ax.set_ylabel("Overconfidence (points)"); ygrid(ax); ax.legend(frameon=False, fontsize=7); panel(ax, "b", x=-.2, y=1.03)
ax.set_title("Stated confidence minus accuracy", fontsize=8.3, loc="left")
fig.savefig("fig_policy.pdf"); plt.close(fig)

# ===================================================================== 5 and 6. audit dumbbells
def dumbbell(ax, labels, truth, est, colors, title):
    y = np.arange(len(labels))[::-1]
    for yi, t, e, c in zip(y, truth, est, colors):
        ax.annotate("", xy=(e, yi), xytext=(t, yi), arrowprops=dict(arrowstyle="-|>", color=c, lw=1.4, shrinkA=4, shrinkB=4), zorder=3)
        ax.plot(t, yi, "o", mfc="white", mec=INK, ms=6.5, mew=1.1, zorder=4); ax.plot(e, yi, "o", color=c, ms=6.5, zorder=4)
        ax.text(e + (.025 if e >= t else -.025), yi + .28, f"{(e - t) * 100:+.0f}", ha="center", fontsize=7.5, color=c, fontweight="bold")
    ax.set_yticks(y); ax.set_yticklabels(labels); ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
    ax.set_xlim(0, 1.06); ax.set_ylim(-.7, len(labels) - .3); xgrid(ax); ax.set_title(title, loc="left")
leg = [Line2D([], [], marker="o", ls="", mfc="white", mec=INK, ms=6.5, label="true value"),
       Line2D([], [], marker="o", ls="", color=MID, ms=6.5, label="label-free estimate")]
fig, ax = plt.subplots(figsize=(4.6, 2.2))
dumbbell(ax, ["Infrared, now", "Visible, now", "Infrared, +0.5 s", "Visible, +0.5 s"], [.901, .798, .895, .798], [.983, .947, .980, .980],
         [IR, VIS, IR, VIS], "Sensitivity, Anti-UAV RGB-T")
ax.set_xlim(.7, 1.03); ax.set_xlabel("Sensitivity"); ax.legend(handles=leg, frameon=False, loc="lower left", fontsize=7.5)
fig.savefig("fig_realaudit.pdf"); plt.close(fig)

fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.7), gridspec_kw=dict(width_ratios=[1.15, 1]))
dumbbell(axs[0], ["Infrared sensitivity", "Infrared specificity", "Visible sensitivity", "Visible specificity"],
         [.854, .454, .354, .762], [.944, .353, .928, .996], [IR, IR, VIS, VIS], "Label-free audit with real negatives")
axs[0].set_xlabel("Value"); axs[0].legend(handles=leg, frameon=False, loc="lower left", fontsize=7.2); panel(axs[0], "a", x=-.38)
ax = axs[1]; names3 = ["Independent\nover time", "Design-effect\ncorrected", "Segment\nmean"]; x = np.arange(3)
over = np.array([28.7, 10.3, 5.2]); lo = np.array([13.3, -5.5, -12.1]); hi = np.array([48.6, 29.6, 24.6])
ax.bar(x, over, .55, color=["#4d4d4d", VIS, "#9ecae1"], zorder=3)
ax.errorbar(x, over, yerr=[over - lo, hi - over], fmt="none", ecolor=INK, elinewidth=1, capsize=4, zorder=4)
for xi, v in zip(x, over): ax.text(xi + .32, v, f"{v:.0f}", va="center", fontsize=7.5)
ax.axhline(0, color=INK, lw=.7); ax.set_xticks(x); ax.set_xticklabels(names3, fontsize=7.8)
ax.set_ylabel("Overconfidence (points)"); ygrid(ax)
ax.set_title("Fusing 25-frame track segments", loc="left"); panel(ax, "b")
ax.text(.98, .97, "bars: 95% video-level\nbootstrap intervals", transform=ax.transAxes, ha="right", va="top", fontsize=6.8, color=MID)
fig.tight_layout(w_pad=2.5); fig.savefig("fig_avid.pdf"); plt.close(fig)

# ===================================================================== 7. error-structure schematic (with icons)
fig = plt.figure(figsize=(6.4, 3.0)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(-7, 45); ax.axis("off")
def box(x, y, w, h, fc, ec):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.35,rounding_size=0.9", fc=fc, ec=ec, lw=1.1))
def drone(x, y, s=.55, c=INK):
    for dy in (-1, 1): ax.plot([x - 1.6 * s, x + 1.6 * s], [y + dy * s, y - dy * s], color=c, lw=.9)
    for dx, dy in ((-1.6, -1), (1.6, 1), (-1.6, 1), (1.6, -1)): ax.add_patch(Circle((x + dx * s, y + dy * s), .72 * s, fill=False, ec=c, lw=.7))
    ax.add_patch(Circle((x, y), .4 * s, color=c))
def sun(x, y):
    ax.add_patch(Circle((x, y), 1.1, color="#E69F00"))
    for a in np.linspace(0, 2 * np.pi, 9)[:-1]: ax.plot([x + 1.6 * np.cos(a), x + 2.3 * np.cos(a)], [y + 1.6 * np.sin(a), y + 2.3 * np.sin(a)], color="#E69F00", lw=.9)
box(30, 35, 40, 6.5, "#fbe3d6", "#c46a3a"); sun(34.2, 38.25)
ax.text(52, 39.6, r"Recording session $k$:  shared effect $\sigma\,v_k$", ha="center", fontsize=8.6, fontweight="bold")
ax.text(52, 36.6, "weather, light, background, range", ha="center", fontsize=7.5, color="#7a4a2a")
for x in (5, 38.5, 72):
    box(x, 21, 23, 7, "#dbe9f6", VIS); drone(x + 3.2, 24.5)
    ax.text(x + 14, 26.1, r"Track $i$:  $\tau\,u_{ik}$", ha="center", fontsize=8.2, fontweight="bold")
    ax.text(x + 14, 23.0, "this target is hard", ha="center", fontsize=7.2, color="#1f4e79")
    ax.add_patch(FancyArrowPatch((50, 34.6), (x + 11.5, 28.4), arrowstyle="-|>", mutation_scale=8, color=MID, lw=.9))
    for fx in (x - .5, x + 7.75, x + 16):
        ax.add_patch(FancyArrowPatch((x + 11.5, 20.6), (fx + 3.5, 14.3), arrowstyle="-|>", mutation_scale=6, color=LIGHT, lw=.8))
        for yy, c, lab in ((10.2, IR, "IR"), (3.4, VIS, "Vis")):
            ax.add_patch(FancyBboxPatch((fx, yy), 7, 3.8, boxstyle="round,pad=0.2,rounding_size=0.6", fc="white", ec=c, lw=1.1))
            ax.text(fx + 3.5, yy + 1.9, lab, ha="center", va="center", fontsize=7, color=c, fontweight="bold")
        ax.plot([fx + 3.5, fx + 3.5], [7.5, 9.9], color=INK, lw=1.8)
ax.text(1.0, 8.6, "frames", fontsize=7.5, rotation=90, va="center", color=MID)
for xx, yy, txt, c in ((90.5, 38.25, r"$\sigma$", "#c46a3a"), (97.5, 24.5, r"$\tau$", VIS), (97.9, 8.0, "OR", INK)):
    ax.add_patch(Circle((xx, yy), 2.1, fc="white", ec=c, lw=1.1)); ax.text(xx, yy, txt, ha="center", va="center", fontsize=8.5, color=c, fontweight="bold")
ax.text(50, -5.2, "Three levels of dependence: shared session conditions ($\\sigma$), a persistently hard track ($\\tau$),\nand infrared and visible failing on the same frame (odds ratio, black links)",
        ha="center", fontsize=7.4, color=INK, linespacing=1.3)
fig.savefig("fig_structure.pdf"); fig.savefig("fig_structure.png", dpi=150); plt.close(fig)

# ===================================================================== 8. lag decay of error dependence (AVID)
fits = [("Infrared misses", IR, "-", .26, .40, 17.6, [(1, .69), (25, .38), (180, .32)]),
        ("Infrared false alarms", IR, (0, (3, 2)), .30, .34, 16.4, [(1, .65), (25, .39), (180, .38)]),
        ("Visible misses", VIS, "-", .00, .94, 160.2, [(1, .92), (25, .81), (180, .25)]),
        ("Visible false alarms", VIS, (0, (3, 2)), .86, .09, 500.0, [(1, .93), (25, .95), (180, .91)])]
fig, ax = plt.subplots(figsize=(5.2, 2.7)); k = np.linspace(1, 190, 400)
kk = k / 25; a_, b_, L_ = .26, .40, 17.6
ax.fill_between(kk, 0, a_, color=IR, alpha=.07, lw=0); ax.fill_between(kk, a_, a_ + b_ * np.exp(-k / L_), color=IR, alpha=.16, lw=0)
ax.text(5.6, .13, "lasting track effect", fontsize=7.2, color="#9a3d00"); ax.annotate("short-lived part\n(fades in ~0.7 s)", xy=(.3, .42), xytext=(.95, .56), fontsize=7, color="#9a3d00", arrowprops=dict(arrowstyle="->", color="#9a3d00", lw=.7))
for lab, c, ls, a, b, L, pts in fits:
    ax.plot(k / 25, a + b * np.exp(-k / L), color=c, ls=ls, lw=1.6, label=lab)
    ax.plot([p[0] / 25 for p in pts], [p[1] for p in pts], "o" if ls == "-" else "s", color=c, ms=4.5, mec="white", mew=.6,
            mfc=c if ls == "-" else "white", zorder=4)
ax.axhspan(0, 0, color="none"); ygrid(ax); ax.set_ylim(0, 1.02); ax.set_xlim(0, 7.6)
ax.set_xlabel("Time between two frames of the same track (seconds)"); ax.set_ylabel("Correlation of the two frames' errors")
ax.legend(frameon=False, fontsize=7.5, loc="lower center", bbox_to_anchor=(.5, 1.0), ncol=2)
ax.annotate("an equal-correlation (random-intercept)\nmodel assumes these lines are flat", xy=(3.3, .36), xytext=(4.1, .66), fontsize=7.2, color=MID,
            arrowprops=dict(arrowstyle="->", color=MID, lw=.7))
fig.savefig("fig_lag.pdf"); plt.close(fig); print("lag figure written")
