"""
Generate method figures for the four window-based sampling approaches (A–D).
Run from the repo root:  python docs/generate_method_figures.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from collections import Counter

OUT = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT, exist_ok=True)

# ── Shared example (same 3×3 window used in every figure) ─────────────────
MAP_PATCH    = np.array([[1, 1, 1],
                          [1, 1, 0],
                          [0, 0, 0]])

INTERP_PATCH = np.array([[1, 0, 0],
                          [1, 1, 0],
                          [0, 0, 1]])

W = 3
pairs       = [(int(MAP_PATCH[i, j]), int(INTERP_PATCH[i, j]))
               for i in range(W) for j in range(W)]
pair_counts = Counter(pairs)          # (0,0):3  (0,1):1  (1,0):2  (1,1):3
prop_map    = float(MAP_PATCH.mean())    # 5/9 ≈ 0.56
prop_interp = float(INTERP_PATCH.mean())  # 4/9 ≈ 0.44
maj_map     = int(prop_map    >= 0.5)   # 1
maj_interp  = int(prop_interp >= 0.5)   # 0
DOMINANT    = (1, 1)   # tie (0,0)=3 & (1,1)=3 → tiebreak prefers (1,1)

# ── Style constants ────────────────────────────────────────────────────────
C1        = "#c94a0b"   # disturbed / class 1 — brick red
C0        = "#ddeeff"   # undisturbed / class 0 — pale blue
HIGHLIGHT = "#fdb462"   # accent cell — amber
EDGE      = "#888888"
LABEL_FS  = 9
CELL_FS   = 12
TITLE_FS  = 10
DPI       = 200

plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.spines.bottom": False,
    "axes.spines.left":   False,
})


# ── Drawing helpers ────────────────────────────────────────────────────────

def draw_grid(ax, patch, title, highlight_cells=None):
    """Draw a W×W binary patch as coloured rounded cells."""
    ax.set_xlim(0, W); ax.set_ylim(0, W)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(title, fontsize=TITLE_FS, fontweight="bold", pad=5)
    if highlight_cells is None:
        highlight_cells = set()

    for i in range(W):
        for j in range(W):
            val = int(patch[i, j])
            color = HIGHLIGHT if (i, j) in highlight_cells else (C1 if val else C0)
            rect = mpatches.FancyBboxPatch(
                (j + 0.07, W - i - 1 + 0.07), 0.86, 0.86,
                boxstyle="round,pad=0.05",
                facecolor=color, edgecolor=EDGE, linewidth=1.2,
            )
            ax.add_patch(rect)
            txt_color = "white" if val else "#444444"
            ax.text(j + 0.5, W - i - 0.5, str(val),
                    ha="center", va="center",
                    fontsize=CELL_FS, fontweight="bold", color=txt_color)


def draw_confusion(ax, n_ij, highlight_cell=None, title="Confusion matrix"):
    """Draw a 2×2 confusion matrix with optional highlighted cell."""
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(title, fontsize=LABEL_FS, pad=4)

    xs = [0.38, 0.62]   # col centres  (ref 0, ref 1)
    ys = [0.62, 0.38]   # row centres  (map 0, map 1)

    # Header labels
    ax.text(0.50, 0.88, "Reference label", ha="center", va="center",
            fontsize=LABEL_FS - 1, fontstyle="italic")
    ax.text(xs[0], 0.78, "0", ha="center", fontsize=LABEL_FS, fontweight="bold")
    ax.text(xs[1], 0.78, "1", ha="center", fontsize=LABEL_FS, fontweight="bold")
    ax.text(0.12, 0.50, "Map\nlabel", ha="center", va="center",
            fontsize=LABEL_FS - 1, fontstyle="italic", rotation=90)
    ax.text(0.25, ys[0], "0", ha="center", fontsize=LABEL_FS, fontweight="bold")
    ax.text(0.25, ys[1], "1", ha="center", fontsize=LABEL_FS, fontweight="bold")

    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            is_hi = (highlight_cell == (i, j))
            rect = mpatches.FancyBboxPatch(
                (x - 0.115, y - 0.115), 0.23, 0.23,
                boxstyle="round,pad=0.02",
                facecolor=HIGHLIGHT if is_hi else "#f4f4f4",
                edgecolor=EDGE, linewidth=1.4,
            )
            ax.add_patch(rect)
            val = n_ij[i, j]
            txt = f"{val:.2f}" if isinstance(val, float) else str(int(val))
            ax.text(x, y, txt, ha="center", va="center",
                    fontsize=CELL_FS - 1,
                    fontweight="bold" if is_hi else "normal")


def add_arrow(fig, ax_left, ax_right, label="", color="#333333"):
    """Draw a horizontal annotation arrow between two axes in figure coords."""
    # We use annotate with fig-level coordinates via transform hack
    x0 = ax_left.get_position().x1 + 0.005
    x1 = ax_right.get_position().x0 - 0.005
    y  = (ax_left.get_position().y0 + ax_left.get_position().y1) / 2
    fig.annotate("", xy=(x1, y), xytext=(x0, y),
                 xycoords="figure fraction", textcoords="figure fraction",
                 arrowprops=dict(arrowstyle="->", color=color, lw=1.8))
    if label:
        fig.text((x0 + x1) / 2, y + 0.015, label,
                 ha="center", va="bottom", fontsize=8, color=color)


# ══════════════════════════════════════════════════════════════════════════
# APPROACH A — every pixel is a sample
# ══════════════════════════════════════════════════════════════════════════
def make_A():
    fig = plt.figure(figsize=(9, 3.2))
    gs  = GridSpec(1, 4, figure=fig,
                   left=0.03, right=0.97, top=0.82, bottom=0.08,
                   wspace=0.35)

    ax_map   = fig.add_subplot(gs[0])
    ax_interp = fig.add_subplot(gs[1])
    ax_pairs  = fig.add_subplot(gs[2])
    ax_cm     = fig.add_subplot(gs[3])

    draw_grid(ax_map,   MAP_PATCH,   "Map field (binary)")
    draw_grid(ax_interp, INTERP_PATCH, "Interpreter field\n(thresholded ≥ 50%)")

    # Pixel-pair listing
    ax_pairs.axis("off")
    ax_pairs.set_title("Per-pixel pairs\n(map, ref)", fontsize=LABEL_FS, pad=5)
    label_map = {(0,0): "TN", (0,1): "FN", (1,0): "FP", (1,1): "TP"}
    lines = []
    for i in range(W):
        for j in range(W):
            p = (int(MAP_PATCH[i,j]), int(INTERP_PATCH[i,j]))
            lines.append(f"({p[0]},{p[1]})  {label_map[p]}")
    ax_pairs.text(0.5, 0.97, "\n".join(lines),
                  ha="center", va="top", fontsize=8,
                  transform=ax_pairs.transAxes,
                  fontfamily="monospace",
                  bbox=dict(boxstyle="round,pad=0.4", facecolor="#f7f7f7", edgecolor=EDGE))

    # Confusion matrix with all 9 counts
    n_ij = np.array([[pair_counts[(0,0)], pair_counts[(0,1)]],
                     [pair_counts[(1,0)], pair_counts[(1,1)]]])
    draw_confusion(ax_cm, n_ij, title="Confusion matrix\n(9 samples total)")

    fig.suptitle("Approach A — pixel-level enumeration", fontsize=12, fontweight="bold", y=0.97)
    plt.savefig(os.path.join(OUT, "approach_A.png"), dpi=DPI, bbox_inches="tight")
    plt.close()
    print("Saved approach_A.png")


# ══════════════════════════════════════════════════════════════════════════
# APPROACH B — dominant pixel-pair per window
# ══════════════════════════════════════════════════════════════════════════
def make_B():
    fig = plt.figure(figsize=(10, 3.2))
    gs  = GridSpec(1, 4, figure=fig,
                   left=0.03, right=0.97, top=0.82, bottom=0.08,
                   wspace=0.38)

    ax_map   = fig.add_subplot(gs[0])
    ax_interp = fig.add_subplot(gs[1])
    ax_counts = fig.add_subplot(gs[2])
    ax_cm     = fig.add_subplot(gs[3])

    draw_grid(ax_map,    MAP_PATCH,    "Map field (binary)")
    draw_grid(ax_interp, INTERP_PATCH, "Interpreter field\n(thresholded ≥ 50%)")

    # Count table
    ax_counts.axis("off")
    ax_counts.set_title("Pixel-pair counts", fontsize=LABEL_FS, pad=5)
    labels_map = {(0,0): "TN", (0,1): "FN", (1,0): "FP", (1,1): "TP"}
    lines = []
    for p in [(0,0),(0,1),(1,0),(1,1)]:
        n = pair_counts[p]
        star = " ◀ dominant" if p == DOMINANT else ""
        lines.append(f"({p[0]},{p[1]}) {labels_map[p]:2s}  n={n}{star}")
    ax_counts.text(0.5, 0.97, "\n".join(lines),
                   ha="center", va="top", fontsize=8,
                   transform=ax_counts.transAxes,
                   fontfamily="monospace",
                   bbox=dict(boxstyle="round,pad=0.4", facecolor="#f7f7f7", edgecolor=EDGE))

    # Confusion matrix — ONE entry at dominant cell
    n_ij = np.zeros((2,2), dtype=int)
    n_ij[DOMINANT[0], DOMINANT[1]] = 1
    draw_confusion(ax_cm, n_ij, highlight_cell=DOMINANT,
                   title="Confusion matrix\n(1 sample — dominant cell)")

    fig.suptitle("Approach B — dominant pixel-pair per window", fontsize=12, fontweight="bold", y=0.97)
    plt.savefig(os.path.join(OUT, "approach_B.png"), dpi=DPI, bbox_inches="tight")
    plt.close()
    print("Saved approach_B.png")


# ══════════════════════════════════════════════════════════════════════════
# APPROACH C — independent majority label per field
# ══════════════════════════════════════════════════════════════════════════
def make_C():
    fig = plt.figure(figsize=(10, 3.4))
    gs  = GridSpec(1, 4, figure=fig,
                   left=0.03, right=0.97, top=0.82, bottom=0.08,
                   wspace=0.38)

    ax_map    = fig.add_subplot(gs[0])
    ax_interp  = fig.add_subplot(gs[1])
    ax_maj     = fig.add_subplot(gs[2])
    ax_cm      = fig.add_subplot(gs[3])

    draw_grid(ax_map,    MAP_PATCH,    "Map field (binary)")
    draw_grid(ax_interp, INTERP_PATCH, "Interpreter field\n(thresholded ≥ 50%)")

    # Majority summary
    ax_maj.axis("off")
    ax_maj.set_title("Majority label\nper field", fontsize=LABEL_FS, pad=5)
    n1_map    = int(MAP_PATCH.sum())
    n1_interp = int(INTERP_PATCH.sum())
    txt = (
        f"Map field:\n"
        f"  {n1_map}/{W*W} pixels = 1\n"
        f"  {n1_map/(W*W):.2f} ≥ 0.5  →  majority = {maj_map}\n\n"
        f"Interpreter field:\n"
        f"  {n1_interp}/{W*W} pixels = 1\n"
        f"  {n1_interp/(W*W):.2f} < 0.5  →  majority = {maj_interp}\n\n"
        f"Window records:  ({maj_map}, {maj_interp})"
    )
    ax_maj.text(0.5, 0.97, txt,
                ha="center", va="top", fontsize=8,
                transform=ax_maj.transAxes,
                fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#f7f7f7", edgecolor=EDGE))

    # Confusion matrix — ONE entry at (maj_map, maj_interp)
    n_ij = np.zeros((2,2), dtype=int)
    n_ij[maj_map, maj_interp] = 1
    draw_confusion(ax_cm, n_ij, highlight_cell=(maj_map, maj_interp),
                   title="Confusion matrix\n(1 sample — majority pair)")

    fig.suptitle("Approach C — independent majority label per field", fontsize=12, fontweight="bold", y=0.97)
    plt.savefig(os.path.join(OUT, "approach_C.png"), dpi=DPI, bbox_inches="tight")
    plt.close()
    print("Saved approach_C.png")


# ══════════════════════════════════════════════════════════════════════════
# APPROACH D — proportional agreement scatter
# ══════════════════════════════════════════════════════════════════════════
def make_D():
    fig = plt.figure(figsize=(10, 3.4))
    gs  = GridSpec(1, 4, figure=fig,
                   left=0.03, right=0.97, top=0.82, bottom=0.08,
                   wspace=0.38)

    ax_map    = fig.add_subplot(gs[0])
    ax_interp  = fig.add_subplot(gs[1])
    ax_props   = fig.add_subplot(gs[2])
    ax_scatter = fig.add_subplot(gs[3])

    draw_grid(ax_map,    MAP_PATCH,    "Map field (binary)")
    draw_grid(ax_interp, INTERP_PATCH, "Interpreter field\n(thresholded ≥ 50%)")

    # Proportion summary
    ax_props.axis("off")
    ax_props.set_title("Proportions\nper field", fontsize=LABEL_FS, pad=5)
    n1_map    = int(MAP_PATCH.sum())
    n1_interp = int(INTERP_PATCH.sum())
    txt = (
        f"Map field:\n"
        f"  {n1_map} / {W*W} = {prop_map:.3f}\n"
        f"  prop_map = {prop_map:.2f}\n\n"
        f"Interpreter field:\n"
        f"  {n1_interp} / {W*W} = {prop_interp:.3f}\n"
        f"  prop_interp = {prop_interp:.2f}\n\n"
        f"Window point:\n"
        f"  ({prop_map:.2f},  {prop_interp:.2f})"
    )
    ax_props.text(0.5, 0.97, txt,
                  ha="center", va="top", fontsize=8,
                  transform=ax_props.transAxes,
                  fontfamily="monospace",
                  bbox=dict(boxstyle="round,pad=0.4", facecolor="#f7f7f7", edgecolor=EDGE))

    # Scatter plot panel (simulate multiple windows)
    rng = np.random.default_rng(42)
    n_fake = 60
    fake_x = np.clip(rng.normal(0.45, 0.22, n_fake), 0, 1)
    fake_y = np.clip(fake_x + rng.normal(0, 0.12, n_fake), 0, 1)

    ax_scatter.scatter(fake_x, fake_y, s=18, alpha=0.35, color="#6baed6", zorder=2)
    ax_scatter.plot([0, 1], [0, 1], "k--", lw=1.2, zorder=3, label="1:1")
    ax_scatter.scatter([prop_map], [prop_interp], s=80, color=C1,
                       zorder=5, label="this window")
    ax_scatter.annotate(f"  ({prop_map:.2f}, {prop_interp:.2f})",
                        xy=(prop_map, prop_interp),
                        fontsize=7.5, color=C1, va="bottom")
    ax_scatter.set_xlim(0, 1); ax_scatter.set_ylim(0, 1)
    ax_scatter.set_xlabel("prop_map", fontsize=8)
    ax_scatter.set_ylabel("prop_interp", fontsize=8)
    ax_scatter.set_title("Agreement scatter", fontsize=LABEL_FS, pad=5)
    ax_scatter.set_aspect("equal")
    ax_scatter.spines["bottom"].set_visible(True)
    ax_scatter.spines["left"].set_visible(True)
    ax_scatter.tick_params(labelsize=7)
    ax_scatter.legend(fontsize=7, loc="upper left")

    fig.suptitle("Approach D — proportional agreement scatter", fontsize=12, fontweight="bold", y=0.97)
    plt.savefig(os.path.join(OUT, "approach_D.png"), dpi=DPI, bbox_inches="tight")
    plt.close()
    print("Saved approach_D.png")


if __name__ == "__main__":
    make_A()
    make_B()
    make_C()
    make_D()
    print("All figures saved to", OUT)
