"""Publication style shared by every main-text flagship figure (2026-08-08 rebuild).

Contract (applies to figs 1,3,4,5,6,7 = fig_puzzle, fig1_gap, fig14_access, fig_dev_scale,
fig_sftrlvr, fig11_devclock/fig12_rlreadout):

  * Figures are drawn AT FINAL SIZE. flagship.tex has \\textwidth = 397.485pt = 5.50 in
    (measured by compiling a probe doc against iclr2027_conference.sty), so a figure included
    at 0.8\\linewidth is designed 4.40 in wide and every point size below is a true point
    size on the page. Nothing is scaled down after the fact.
  * One font family, one size scale: 7 pt body, 6.5 pt annotations/ticks, 8 pt bold panel
    letters. Floor is 6.5 pt >= the 5 pt rendered-glyph floor.
  * White background everywhere (the old cream PAPER tint is retired for data panels).
  * One semantic palette across ALL figures -- a colour always means the same thing:
      TEAL  representation / probe / modern recipe
      CLAY  behaviour / pre-2023 recipe / erosion
      OCHRE behaviour after supplying the readout (steering / repair)
      GREY  nulls, baselines, censoring
    The teal/clay pair is the house pair already CVD-validated for this paper.
  * Direct labels are preferred to legend boxes; a legend is used only where three or more
    interleaved series make direct labels ambiguous, and then it is frameless.
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt  # noqa: F401  (re-exported)

INK = "#1A1A1A"
TEAL = "#1F6F78"    # representation / probe / modern recipe
CLAY = "#C0442E"    # behaviour / pre-2023 recipe / erosion
OCHRE = "#B8860B"   # steered / repair
GREY = "#8A8F94"    # nulls, baselines, censored
GRID = "#DDDDDD"

# lightness ramps for the scale-overlay figure (dash pattern is the primary identity channel)
TEAL_RAMP = ["#9EC7CC", "#73AEB5", "#47949E", "#1F6F78", "#14595F"]
CLAY_RAMP = ["#D98C70", "#C4613E", "#993620"]

BODY, SMALL, PANEL = 7.0, 6.5, 8.0
# Mathtext renders sub/superscripts at ~0.7x the parent size, so any label carrying a script
# (subscripts, log-tick exponents) must use a parent >= 7.2 pt to keep every rendered glyph
# at or above the 5 pt floor (0.7 * 7.2 = 5.04). Verified with the exported-PDF Tf audit.
MATH = 7.2


def house() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "Helvetica Neue", "DejaVu Sans"],
        "font.size": BODY,
        "pdf.fonttype": 42, "ps.fonttype": 42,   # Type 42, never Type 3 (verify.py gate)
        "svg.fonttype": "none",                  # editable text if an SVG is ever exported
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.6, "axes.edgecolor": "#444444",
        "axes.labelsize": BODY, "axes.labelcolor": INK,
        "xtick.labelsize": SMALL, "ytick.labelsize": SMALL,
        "xtick.color": "#444444", "ytick.color": "#444444",
        "xtick.major.size": 2.6, "ytick.major.size": 2.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "legend.frameon": False, "legend.fontsize": SMALL,
        "text.color": INK,
        "lines.solid_capstyle": "round",
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })


def panel_title(ax, letter: str, finding: str) -> None:
    """Bold lowercase panel letter + the finding as the title, left-aligned (Nature style)."""
    ax.set_title(f"$\\bf{{{letter}}}$  {finding}", fontsize=BODY, loc="left", pad=4)


def ygrid(ax) -> None:
    ax.grid(axis="y", color=GRID, lw=0.4, zorder=0)
    ax.set_axisbelow(True)


def line_label(ax, x, y, text, color=GREY, ha="right", side="above", fontsize=SMALL):
    """Direct label for a reference line. The offset is in POINTS, not data units, so the
    clearance from the line survives any axis rescaling (audit() enforces it)."""
    dy = 3.2 if side == "above" else -3.2
    ax.annotate(text, xy=(x, y), xytext=(0, dy), textcoords="offset points", ha=ha,
                va="bottom" if side == "above" else "top",
                fontsize=fontsize, color=color, zorder=4)


def logx(ax):
    """Log x-axis whose 10^n tick exponents stay above the 5 pt glyph floor."""
    ax.set_xscale("log")
    ax.tick_params(axis="x", labelsize=MATH)


def audit(fig, name):
    """Pixel-space collision audit: every text/legend bounding box against every rendered
    line (segments densified in display coords, so diagonal segments are covered) and every
    bar/patch, plus text-vs-text. Texts marked gid='deliberate' (e.g. white labels drawn
    INSIDE a bar on purpose) are exempt. Returns the number of collisions found; the caller
    treats a nonzero count as a failed gate. Eyeballing missed exactly these; this does not.
    """
    import numpy as np
    from matplotlib.text import Text

    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    PAD = 1.5  # px; also covers small line markers

    def text_bbox(t):
        # Annotation.get_window_extent unions in the ARROW, and an arrow legitimately points
        # into the data -- audit the text box alone.
        return Text.get_window_extent(t, rend)

    texts = []
    for t in fig.texts:
        if t.get_visible() and t.get_text().strip() and t.get_gid() != "deliberate":
            texts.append((f"figtext {t.get_text()[:36]!r}", text_bbox(t)))
    for lg in fig.legends:
        texts.append(("fig-legend", lg.get_window_extent(rend)))
    for ax in fig.axes:
        for t in ax.texts:
            if t.get_visible() and t.get_text().strip() and t.get_gid() != "deliberate":
                texts.append((f"text {t.get_text()[:36]!r}", text_bbox(t)))
        if ax.get_legend() is not None:
            texts.append(("legend", ax.get_legend().get_window_extent(rend)))

    # tick labels and axis labels are obstacles too: an annotation drifting into the tick or
    # label zone is a collision
    ticks = []
    for ax in fig.axes:
        for t in (list(ax.get_xticklabels()) + list(ax.get_yticklabels())
                  + [ax.xaxis.label, ax.yaxis.label]):
            if t.get_visible() and t.get_text().strip():
                ticks.append(t.get_window_extent(rend))

    issues = []
    for label, bb in texts:
        box = bb.padded(PAD)
        hit = None
        for tb in ticks:
            if tb.padded(1).overlaps(bb):
                hit = "a tick label"
                break
        if hit:
            issues.append(f"  COLLISION [{name}]: {label} vs {hit}")
            continue
        for ax in fig.axes:
            for ln in ax.lines:
                x, y = ln.get_data()
                pts = ln.get_transform().transform(np.column_stack([np.asarray(x, float),
                                                                    np.asarray(y, float)]))
                pts = pts[np.isfinite(pts).all(axis=1)]
                if len(pts) > 1:  # densify so diagonal segments cannot pass through unseen
                    f = np.linspace(0, 1, 24)[:, None]
                    seg = pts[:-1][None] * (1 - f[:, :, None]) + pts[1:][None] * f[:, :, None]
                    pts = seg.reshape(-1, 2)
                if len(pts) and ((pts[:, 0] >= box.x0) & (pts[:, 0] <= box.x1)
                                 & (pts[:, 1] >= box.y0) & (pts[:, 1] <= box.y1)).any():
                    hit = f"a line ({ln.get_color()})"
                    break
            if hit is None:
                for p in ax.patches:
                    al = p.get_alpha()
                    if al is not None and al <= 0.3:
                        continue  # a background wash (axvspan zone): annotating over it is fine
                    pb = p.get_window_extent(rend)
                    if pb.width > 0 and pb.height > 0 and pb.padded(PAD).overlaps(bb):
                        hit = "a bar/patch"
                        break
            if hit:
                issues.append(f"  COLLISION [{name}]: {label} vs {hit}")
                break
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            if texts[i][1].padded(1).overlaps(texts[j][1]):
                issues.append(f"  COLLISION [{name}]: {texts[i][0]} vs {texts[j][0]}")
    for line in issues:
        print(line)
    if not issues:
        print(f"  audit [{name}]: no collisions")
    return len(issues)
