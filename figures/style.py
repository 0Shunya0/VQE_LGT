"""
Shared matplotlib presentation style for all figures/make_figN.py scripts.

Presentation only -- no data, computation, or numerical values live here.
Import and call apply() before building any figure; use panel_label() in
place of each script's former copy-pasted _panel() helper; use FULL_WIDTH /
COL_WIDTH as the figsize width for full-textwidth vs single-column figures
in the two-column revtex layout.

Figures are included at \\textwidth / \\columnwidth with NO scaling, so the
figure's physical size on the page equals figsize exactly, and the point
sizes below are the point sizes readers see on the page. Body text in the
paper is 10pt, so axis/tick/legend text is kept at or below that.
"""
import matplotlib.pyplot as plt

# \includegraphics width in the two-column revtex document.
FULL_WIDTH = 7.0   # inches -- figure spans both columns (\textwidth)
COL_WIDTH = 3.4    # inches -- figure sits in a single column

RCPARAMS = {
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "lines.linewidth": 1.2,
    "lines.markersize": 4,
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "legend.framealpha": 1.0,
    "legend.frameon": True,
    "axes.labelpad": 2,
    "figure.constrained_layout.use": False,
}


def apply():
    """Apply the shared rcParams. Call once, before creating any figure."""
    plt.rcParams.update(RCPARAMS)


def panel_label(ax, label, fontsize=10, fontweight="bold", y=1.02):
    """Bare (a)/(b)/... subplot label, bold, just above the axes frame.
    Descriptive text lives in the manuscript caption, not on the figure."""
    ax.text(0.0, y, label, transform=ax.transAxes, va="bottom", ha="left",
             fontweight=fontweight, fontsize=fontsize)
