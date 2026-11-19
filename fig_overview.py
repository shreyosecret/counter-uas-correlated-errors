import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch, Polygon, FancyArrowPatch, Wedge
plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8.5})
rng = np.random.default_rng(3)
fig = plt.figure(figsize=(6.6, 4.6)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 70); ax.axis("off")
INK, IR, VIS, GR = "#1a1a1a", "#D55E00", "#0072B2", "#8c8c8c"
# sky and ground
ax.add_patch(Rectangle((0, 24), 100, 46, color="#f3f6fa", zorder=0))
ax.add_patch(Rectangle((0, 22), 100, 2.2, color="#d9d4c7", zorder=1))
# trailer-mounted sensor mast: telescoping mast, pan-tilt head, visible and thermal cameras
from matplotlib.patches import Polygon as PolyM
G0 = 24.2; TR, TR2, MET = "#6f7b62", "#4f5945", "#8f979e"
PM = lambda pts, c, z=4, ec="#30352b", lw=.6, **k: ax.add_patch(PolyM(pts, closed=True, fc=c, ec=ec, lw=lw, zorder=z, **k))
PM([[9.6, G0 + 1.3], [19.4, G0 + 1.3], [19.4, G0 + 2.7], [9.6, G0 + 2.7]], TR)                            # trailer body
PM([[9.6, G0 + 2.7], [19.4, G0 + 2.7], [19.0, G0 + 3.0], [10.0, G0 + 3.0]], TR2)                          # deck lip
ax.plot([9.6, 6.6], [G0 + 1.7, G0 + 1.1], color="#30352b", lw=1.3, zorder=4)                                # tow bar
ax.add_patch(Circle((6.4, G0 + 1.05), .32, fc="#30352b", lw=0, zorder=5))                                   # hitch
for x0, x1 in ((10.2, 9.0), (18.8, 20.0)):                                                                  # stabilizer legs
    ax.plot([x0, x1], [G0 + 1.4, G0 + .15], color="#30352b", lw=1, zorder=4)
    PM([[x1 - .45, G0], [x1 + .45, G0], [x1 + .45, G0 + .22], [x1 - .45, G0 + .22]], "#30352b", z=5, lw=0)
ax.add_patch(Circle((14.5, G0 + 1.25), 1.25, fc="#262626", ec="#111", lw=.5, zorder=6))                     # wheel
ax.add_patch(Circle((14.5, G0 + 1.25), .62, fc="#7d8070", ec="#3b3d34", lw=.4, zorder=7))
PM([[12.9, G0 + 2.6], [16.1, G0 + 2.6], [16.1, G0 + 3.2], [12.9, G0 + 3.2]], TR2, z=5)                     # fender
segs = [(15.0, .55, G0 + 3.0, G0 + 4.9), (15.0, .42, G0 + 4.9, G0 + 6.5), (15.0, .3, G0 + 6.5, G0 + 7.7)]
for cx, hw, y0, y1 in segs:                                                                                 # telescoping mast
    PM([[cx - hw, y0], [cx + hw, y0], [cx + hw, y1], [cx - hw, y1]], MET, z=5, ec="#4a4f55", lw=.5)
    ax.plot([cx - hw - .12, cx + hw + .12], [y1, y1], color="#4a4f55", lw=1.2, zorder=6)                    # collar
ax.add_patch(FancyBboxPatch((14.1, G0 + 7.7), 1.8, .9, boxstyle="round,pad=0.08,rounding_size=0.25",
                            fc="#5f666d", ec="#3a3f44", lw=.5, zorder=6))                                    # pan unit
PM([[14.25, G0 + 8.6], [14.75, G0 + 8.6], [14.75, G0 + 13.9], [14.25, G0 + 13.9]], "#5f666d", z=5, ec="#3a3f44", lw=.5)  # tilt bracket
# visible camera (top): housing, sun hood, lens with blue ring
ax.add_patch(FancyBboxPatch((14.8, 35.6), 4.6, 1.8, boxstyle="round,pad=0.06,rounding_size=0.35", fc="#dfe3e8", ec="#4a4f55", lw=.6, zorder=6))
PM([[14.6, 37.45], [20.4, 37.45], [20.4, 37.8], [14.6, 37.8]], "#c4cad1", z=7, ec="#4a4f55", lw=.5)        # sun hood
PM([[19.4, 35.85], [20.0, 35.85], [20.0, 37.15], [19.4, 37.15]], "#3a3f44", z=7, lw=.4)                    # bezel
PM([[20.0, 35.95], [20.55, 35.95], [20.55, 37.05], [20.0, 37.05]], "#1d2730", z=7, ec=VIS, lw=1.1)        # lens
ax.add_patch(Circle((15.6, 36.5), .18, fc=VIS, lw=0, zorder=7))                                              # status light
# thermal camera (bottom): bulkier housing, large germanium lens with orange ring, fins
ax.add_patch(FancyBboxPatch((14.8, 32.5), 4.1, 2.4, boxstyle="round,pad=0.06,rounding_size=0.4", fc="#cfc9c1", ec="#4a4f55", lw=.6, zorder=6))
for yy in (32.95, 33.45, 33.95, 34.45): ax.plot([15.3, 16.6], [yy, yy], color="#9a948c", lw=.6, zorder=7)  # cooling fins
PM([[18.9, 32.7], [19.35, 32.7], [19.35, 34.7], [18.9, 34.7]], IR, z=7, lw=.4)                             # orange ring
PM([[19.35, 32.8], [20.5, 32.95], [20.5, 34.45], [19.35, 34.6]], "#2a2630", z=7, lw=.5)                   # germanium barrel
ax.plot([20.5, 20.5], [33.05, 34.35], color="#6b5b7a", lw=1.6, zorder=8)                                    # lens face sheen
ax.plot([14.75, 14.75], [33.2, 36.2], color="#30352b", lw=.7, zorder=4)                                     # cable
# fields of view from the lenses, and colour-coded labels
for (ox, oy), c, lab, ang, ly in (((20.55, 36.5), VIS, "Visible", (4, 28), 36.5), ((20.5, 33.7), IR, "Thermal IR", (1, 25), 33.7)):
    ax.add_patch(Wedge((ox, oy), 86, ang[0], ang[1], color=c, alpha=0.07, zorder=1))
    ax.text(13.6, ly, lab, ha="right", va="center", color=c, fontsize=7.5, fontweight="bold")
ax.text(13.2, 19.6, "Sensor mast", ha="center", fontsize=7.5)

# ---------------------------------------------------------------- drones, detections and one hard track
from matplotlib.patches import Rectangle as Rect
def quad(x, y, s=1.0, alpha=1.0, z=5):
    for dx, dy in ((-1, -1), (1, 1)):
        ax.plot([x - 1.6 * s, x + 1.6 * s], [y + dy * s, y - dy * s], color=INK, lw=1, zorder=z, alpha=alpha)
    for dx, dy in ((-1.6, -1), (1.6, 1), (-1.6, 1), (1.6, -1)):
        ax.add_patch(Circle((x + dx * s, y + dy * s), 0.75 * s, fill=False, ec=INK, lw=.8, zorder=z, alpha=alpha))
    ax.add_patch(Circle((x, y), .45 * s, color=INK, zorder=z + 1, alpha=alpha))
def boxes(x, y, ir=True, vis=True):
    if ir:  ax.add_patch(Rect((x - 3.1, y - 2.3), 6.2, 4.6, fill=False, ec=IR, lw=1.1, zorder=6))
    if vis: ax.add_patch(Rect((x - 3.7, y - 2.9), 7.4, 5.8, fill=False, ec=VIS, lw=1.0, ls=(0, (3, 1.6)), zorder=6))
seen = [(58, 58), (66, 63.5), (74, 57), (83, 62.5), (91, 55.5), (63, 49.5), (90, 45.5)]
for i, (x, y) in enumerate(seen):
    quad(x, y, .85); boxes(x, y, ir=True, vis=(i != 4))          # one drone: infrared sees it, visible misses
ax.text(91, 51.2, "visible misses,\ninfrared sees", ha="center", fontsize=6.4, color=GR, va="top")
# the hard drone: missed by both cameras, frame after frame (ghosts = earlier queries of the same track)
track = [(66, 38.5), (72.5, 40.5), (79, 42.5)]
ax.plot([p[0] for p in track], [p[1] for p in track], color=GR, lw=.9, ls=(0, (2, 2)), zorder=4)
for (x, y), a, lab in zip(track, (.25, .5, 1), ("$t-2$", "$t-1$", "$t$")):
    quad(x, y, .85, alpha=a); ax.text(x, y - (5.0 if lab == "$t$" else 3.4), lab, ha="center", fontsize=7, color=GR)
ax.add_patch(Circle(track[-1], 3.6, fill=False, ec="#b03a2e", lw=1.2, ls=(0, (2, 1.5)), zorder=6))
ax.text(83.4, 39.0, "missed by both\ncameras, every frame", fontsize=7, color="#b03a2e", va="center")
ax.text(70, 68.4, "Drone swarm: each drone is one track", ha="center", fontsize=8)
# birds (correctly not boxed)
for (x, y) in ((40, 61), (45, 64.5)):
    ax.plot([x - 1.6, x, x + 1.6], [y + .9, y, y + .9], color=GR, lw=1.1)
ax.text(42.5, 58.3, "birds", ha="center", color=GR, fontsize=7)
# annotations tied to the paper's quantities
ax.annotate(r"re-queries of one track repeat" + "\n" + r"the same error ($\tau$)", xy=(69.5, 39.6), xytext=(36, 45.5), fontsize=7.6,
            arrowprops=dict(arrowstyle="->", color=INK, lw=.7))
ax.annotate("IR and visible miss together\n(same-frame odds ratio about 6)", xy=(75.6, 41.6), xytext=(36, 28.6), fontsize=7.6,
            arrowprops=dict(arrowstyle="->", color=INK, lw=.7, connectionstyle="arc3,rad=-.15"))
# legend for the detection boxes
lx, ly = 83, 29.5
ax.add_patch(Rect((lx, ly), 3.2, 2.2, fill=False, ec=IR, lw=1.1)); ax.text(lx + 4.2, ly + 1.1, "infrared detection", va="center", fontsize=7)
ax.add_patch(Rect((lx, ly - 3.6), 3.2, 2.2, fill=False, ec=VIS, lw=1.0, ls=(0, (3, 1.6)))); ax.text(lx + 4.2, ly - 2.5, "visible detection", va="center", fontsize=7)
# ---------------------------------------------------------------- pipeline studied in this paper (full width)
steps = ["Detect", "Verify\n(re-query)", "Fuse\nIR + visible", "Audit\n(latent class)"]
w, gap, x0, y0, h = 21.0, 4.0, 2.5, 3.5, 9
for k, lab in enumerate(steps):
    x = x0 + k * (w + gap)
    ax.add_patch(FancyBboxPatch((x, y0), w, h, boxstyle="round,pad=0.3", fc="#dbe9f6", ec=VIS, lw=1, zorder=3))
    ax.text(x + w / 2, y0 + h / 2, lab, ha="center", va="center", fontsize=7.8, color=INK, zorder=4)
    if k: ax.add_patch(FancyArrowPatch((x - gap - .1, y0 + h / 2), (x + .1, y0 + h / 2), arrowstyle="-|>", mutation_scale=8, color=INK, zorder=4))
ax.text(50, y0 + h + 2.3, "Perception pipeline studied in this paper", ha="center", fontsize=7.8, color=VIS)
fig.savefig("fig_overview.pdf"); fig.savefig("fig_overview.png", dpi=160); print("ok")
