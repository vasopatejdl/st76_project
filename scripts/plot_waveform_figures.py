#!/usr/bin/env python3.11
"""
Generate all three waveform comparison figures for the report,
with a unified visual style.

Usage:
    python3 scripts/plot_waveform_figures.py

Outputs (in Rapport_SEISM/img/):
    fig_trace_norms.png     – pairwise trace-norm comparison (scaled)
    fig_uobs_vs_final.png   – component-wise Uobs vs Sim k=20
    fig_baseline_vs_final.png – component-wise Sim k=0 vs Sim k=20
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import numpy as np
import matplotlib.pyplot as plt
from compare_traces import load_traces

# ─── Global style ───────────────────────────────────────────────
COLOR_OBS  = '#1a1a1a'       # near-black for Uobs
COLOR_K0   = '#c0392b'       # muted red for Sim k=0
COLOR_K20  = '#2980b9'       # steel blue for Sim k=20

LW_PRIMARY = 1.0             # line width for the "primary" trace
LW_SECOND  = 0.9             # line width for the "secondary" trace
ALPHA      = 0.90
GRID_ALPHA = 0.25
TMAX       = 2.0
DPI        = 200
FIGSIZE_3x3 = (12, 7.5)      # 3 station × 3 column grids
TITLE_SIZE = 13
SUBTITLE_SIZE = 11
LABEL_SIZE = 9
TICK_SIZE  = 8

STATIONS = ['Uobs_0000', 'Uobs_0030', 'Uobs_0120']
STATION_LABELS = ['Station 0000', 'Station 0030', 'Station 0120']
COMPONENTS = ['Ux', 'Uy', 'Uz']

plt.rcParams.update({
    'font.size': LABEL_SIZE,
    'axes.titlesize': SUBTITLE_SIZE,
    'axes.labelsize': LABEL_SIZE,
    'xtick.labelsize': TICK_SIZE,
    'ytick.labelsize': TICK_SIZE,
    'legend.fontsize': LABEL_SIZE,
})

# ─── Paths ──────────────────────────────────────────────────────
BASE   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UOBS   = os.path.join(BASE, 'Uobs', 'capteurs.0025.h5')
IT0    = os.path.join(BASE, 'SEM3D_ST7', 'archive', 'it0', 'traces')
IT20   = os.path.join(BASE, 'SEM3D_ST7', 'archive', 'it20', 'traces')
OUTDIR = os.path.join(BASE, 'Rapport_SEISM', 'img')
os.makedirs(OUTDIR, exist_ok=True)

# ─── Load data ──────────────────────────────────────────────────
print('Loading traces …')
u_t, u_d, _ = load_traces(UOBS)
k0_t, k0_d, _ = load_traces(IT0)
kf_t, kf_d, _ = load_traces(IT20)


def _grid_axes(fig, axes):
    """Common grid/axis formatting."""
    for row in axes:
        for ax in row:
            ax.set_xlim(0, TMAX)
            ax.grid(True, alpha=GRID_ALPHA)


def _add_legend(fig, handles, labels, ncol=3):
    fig.legend(handles, labels, loc='upper center',
               bbox_to_anchor=(0.5, 0.995), ncol=ncol,
               frameon=False, fontsize=LABEL_SIZE + 1)


# ═══════════════════════════════════════════════════════════════
#  Figure 1 – Trace norms (pairwise)
# ═══════════════════════════════════════════════════════════════
def make_trace_norms():
    # Compute norms on common time (uobs time base)
    norms = {}
    for st in STATIONS:
        u = np.linalg.norm(u_d[st], axis=1)
        s0 = np.linalg.norm(np.column_stack(
            [np.interp(u_t, k0_t, k0_d[st][:, i], left=0, right=0) for i in range(3)]
        ), axis=1)
        sf = np.linalg.norm(np.column_stack(
            [np.interp(u_t, kf_t, kf_d[st][:, i], left=0, right=0) for i in range(3)]
        ), axis=1)
        # Per-station scaling by Uobs peak
        scale = max(np.max(np.abs(u)), 1e-12)
        norms[st] = [u / scale, s0 / scale, sf / scale]

    pairs = [
        ('Uobs', 'Sim $k$=0', 0, 1, COLOR_OBS, COLOR_K0),
        ('Uobs', 'Sim $k$=20', 0, 2, COLOR_OBS, COLOR_K20),
        ('Sim $k$=0', 'Sim $k$=20', 1, 2, COLOR_K0, COLOR_K20),
    ]

    fig, axes = plt.subplots(3, 3, figsize=FIGSIZE_3x3, sharex=True, sharey='row')
    for i, (st, stl) in enumerate(zip(STATIONS, STATION_LABELS)):
        for j, (l1, l2, a, b, c1, c2) in enumerate(pairs):
            ax = axes[i, j]
            ax.plot(u_t, norms[st][a], color=c1, lw=LW_PRIMARY, alpha=ALPHA)
            ax.plot(u_t, norms[st][b], color=c2, lw=LW_SECOND, alpha=ALPHA)
            if i == 0:
                ax.set_title(f'{l1}  vs  {l2}')
            if j == 0:
                ax.set_ylabel(f'{stl}\nNorm (scaled)')
            if i == 2:
                ax.set_xlabel('Time (s)')

    _grid_axes(fig, axes)
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=COLOR_OBS, lw=1.5, label='Uobs'),
               Line2D([0], [0], color=COLOR_K0, lw=1.5, label='Sim $k$=0'),
               Line2D([0], [0], color=COLOR_K20, lw=1.5, label='Sim $k$=20')]
    _add_legend(fig, handles, [h.get_label() for h in handles])
    fig.suptitle('Pairwise trace-norm comparison (per-station scaling)',
                 fontsize=TITLE_SIZE, y=1.03)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(OUTDIR, 'fig_trace_norms.png')
    fig.savefig(out, dpi=DPI, bbox_inches='tight')
    print(f'  → {out}')
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════
#  Figure 2 – Uobs vs final (component-wise)
# ═══════════════════════════════════════════════════════════════
def make_uobs_vs_final():
    fig, axes = plt.subplots(3, 3, figsize=FIGSIZE_3x3, sharex=True)
    for i, (st, stl) in enumerate(zip(STATIONS, STATION_LABELS)):
        for j, c in enumerate(COMPONENTS):
            ax = axes[i, j]
            ax.plot(u_t, u_d[st][:, j], color=COLOR_OBS, lw=LW_PRIMARY, alpha=ALPHA)
            ax.plot(kf_t, kf_d[st][:, j], color=COLOR_K20, lw=LW_SECOND, alpha=ALPHA)
            if i == 0:
                ax.set_title(c)
            if j == 0:
                ax.set_ylabel(f'{stl}\nDispl (m)')
            if i == 2:
                ax.set_xlabel('Time (s)')

    _grid_axes(fig, axes)
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=COLOR_OBS, lw=1.5, label='Uobs'),
               Line2D([0], [0], color=COLOR_K20, lw=1.5, label='Sim $k$=20')]
    _add_legend(fig, handles, [h.get_label() for h in handles], ncol=2)
    fig.suptitle('Observed vs final simulation ($k$=20)',
                 fontsize=TITLE_SIZE, y=1.03)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(OUTDIR, 'fig_uobs_vs_final.png')
    fig.savefig(out, dpi=DPI, bbox_inches='tight')
    print(f'  → {out}')
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════
#  Figure 3 – Baseline vs final (component-wise)
# ═══════════════════════════════════════════════════════════════
def make_baseline_vs_final():
    fig, axes = plt.subplots(3, 3, figsize=FIGSIZE_3x3, sharex=True)
    for i, (st, stl) in enumerate(zip(STATIONS, STATION_LABELS)):
        for j, c in enumerate(COMPONENTS):
            ax = axes[i, j]
            ax.plot(k0_t, k0_d[st][:, j], color=COLOR_K0, lw=LW_PRIMARY, alpha=ALPHA)
            ax.plot(kf_t, kf_d[st][:, j], color=COLOR_K20, lw=LW_SECOND, alpha=ALPHA)
            if i == 0:
                ax.set_title(c)
            if j == 0:
                ax.set_ylabel(f'{stl}\nDispl (m)')
            if i == 2:
                ax.set_xlabel('Time (s)')

    _grid_axes(fig, axes)
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=COLOR_K0, lw=1.5, label='Sim $k$=0'),
               Line2D([0], [0], color=COLOR_K20, lw=1.5, label='Sim $k$=20')]
    _add_legend(fig, handles, [h.get_label() for h in handles], ncol=2)
    fig.suptitle('Baseline ($k$=0) vs final ($k$=20) simulation',
                 fontsize=TITLE_SIZE, y=1.03)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(OUTDIR, 'fig_baseline_vs_final.png')
    fig.savefig(out, dpi=DPI, bbox_inches='tight')
    print(f'  → {out}')
    plt.close(fig)


# ─── Main ───────────────────────────────────────────────────────
if __name__ == '__main__':
    make_trace_norms()
    make_uobs_vs_final()
    make_baseline_vs_final()
    print('Done.')
