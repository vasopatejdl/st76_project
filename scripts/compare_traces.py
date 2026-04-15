#!/usr/bin/env python3.11
"""
Compare simulated traces against Uobs (observed) displacements.

Usage:
    python3 scripts/compare_traces.py 
    python3 scripts/compare_traces.py --sim SEM3D_ST7/traces/capteurs.0065.h5
    python3 scripts/compare_traces.py --stations 0000 0030 0060 0090 0120
    python3 scripts/compare_traces.py --tmax 1.0 --output my_comparison.png
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
import math


def load_traces(filepath, displ_cols=(2, 3, 4)):
    """Load displacement traces from one or more SEM3D HDF5 capteurs files."""
    import glob, os
    
    if os.path.isdir(filepath):
        files = sorted(glob.glob(os.path.join(filepath, 'capteurs.*.h5')))
    else:
        files = [filepath]
    
    disp = {}
    pos = {}
    time = None
    
    for fpath in files:
        with h5py.File(fpath, 'r') as f:
            keys = sorted(f.keys())
            data_keys = [k for k in keys if not k.endswith('_pos') and k != 'Variables']
            
            for k in data_keys:
                arr = f[k][...]
                if time is None:
                    time = arr[:, 0]
                disp[k] = arr[:, list(displ_cols)]
            
            for k in keys:
                if k.endswith('_pos'):
                    name = k.replace('_pos', '')
                    pos[name] = f[k][...]
    
    return time, disp, pos


def amplitude_stats(disp):
    """Compute per-component amplitude statistics."""
    all_data = np.array(list(disp.values()))  # (n_stations, nt, 3)
    stats = {}
    for i, comp in enumerate(['Ux', 'Uy', 'Uz']):
        d = all_data[:, :, i]
        stats[comp] = {
            'max': np.abs(d).max(),
            'rms': np.sqrt(np.mean(d**2)),
        }
    return stats


def print_comparison(uobs_stats, sim_stats):
    """Print amplitude comparison table."""
    print(f"\n{'Component':<10} {'Uobs max':>12} {'Sim max':>12} {'Ratio':>8}  {'Uobs RMS':>12} {'Sim RMS':>12} {'Ratio':>8}")
    print("-" * 80)
    for comp in ['Ux', 'Uy', 'Uz']:
        u = uobs_stats[comp]
        s = sim_stats[comp]
        max_ratio = u['max'] / s['max'] if s['max'] > 0 else float('inf')
        rms_ratio = u['rms'] / s['rms'] if s['rms'] > 0 else float('inf')
        print(f"{comp:<10} {u['max']:>12.2e} {s['max']:>12.2e} {max_ratio:>8.2f}  {u['rms']:>12.2e} {s['rms']:>12.2e} {rms_ratio:>8.2f}")


def trace_norm(trace):
    """Vector norm of a 3-component displacement trace."""
    return np.linalg.norm(trace, axis=1)


def first_break_time(time, trace, fraction=0.05):
    """Estimate first-arrival time.

    Returns a tuple:
        (first_nonzero_time, threshold_crossing_time)

    The threshold crossing uses a fraction of the trace peak norm, which is
    usually more robust than the very first non-zero sample.
    """
    amp = trace_norm(trace)
    nz = np.flatnonzero(amp > 0.0)
    first_nonzero = time[nz[0]] if len(nz) else math.nan

    peak = amp.max()
    threshold = fraction * peak
    hit = np.flatnonzero(amp >= threshold)
    threshold_crossing = time[hit[0]] if len(hit) else math.nan
    return first_nonzero, threshold_crossing


def resample_trace(time, trace, new_time):
    """Linearly resample a 3-component trace onto a common time axis."""
    return np.column_stack([
        np.interp(new_time, time, trace[:, i])
        for i in range(trace.shape[1])
    ])


def cross_correlation_lag(uobs_time, uobs_trace, sim_time, sim_trace):
    """Estimate time lag from cross-correlation of trace norms.

    Negative lag => simulation arrives earlier than Uobs.
    Positive lag => simulation arrives later than Uobs.
    """
    t0 = max(uobs_time[0], sim_time[0])
    t1 = min(uobs_time[-1], sim_time[-1])
    dt = max(uobs_time[1] - uobs_time[0], sim_time[1] - sim_time[0])
    common_time = np.arange(t0, t1, dt)

    u = trace_norm(resample_trace(uobs_time, uobs_trace, common_time))
    s = trace_norm(resample_trace(sim_time, sim_trace, common_time))
    u = u - u.mean()
    s = s - s.mean()

    cc = np.correlate(s, u, mode='full')
    lags = np.arange(-len(common_time) + 1, len(common_time))
    return lags[np.argmax(cc)] * dt


def print_arrival_comparison(uobs_time, uobs_disp, sim_time, sim_disp, stations, fraction=0.05):
    """Print per-station arrival-time comparison and global summaries."""
    print(f"\nArrival-time comparison (first non-zero and {fraction:.0%} of peak norm)")
    print(f"{'Station':<12} {'Uobs first':>12} {'Sim first':>12} {'Δ first':>12}  {'Uobs thr':>12} {'Sim thr':>12} {'Δ thr':>12}  {'XCorr lag':>12}")
    print("-" * 110)

    all_thr_deltas = []
    all_xcorr_lags = []
    common = sorted(set(uobs_disp) & set(sim_disp))
    for station in common:
        u_first, u_thr = first_break_time(uobs_time, uobs_disp[station], fraction=fraction)
        s_first, s_thr = first_break_time(sim_time, sim_disp[station], fraction=fraction)
        xlag = cross_correlation_lag(uobs_time, uobs_disp[station], sim_time, sim_disp[station])

        all_thr_deltas.append(s_thr - u_thr)
        all_xcorr_lags.append(xlag)

        if station in stations:
            print(f"{station:<12} {u_first:>12.6f} {s_first:>12.6f} {s_first-u_first:>12.6f}  {u_thr:>12.6f} {s_thr:>12.6f} {s_thr-u_thr:>12.6f}  {xlag:>12.6f}")

    all_thr_deltas = np.asarray(all_thr_deltas)
    all_xcorr_lags = np.asarray(all_xcorr_lags)
    print("-" * 110)
    print(f"Threshold-arrival Δt (Sim - Uobs) over all stations: mean={all_thr_deltas.mean():.6f} s, median={np.median(all_thr_deltas):.6f} s, min={all_thr_deltas.min():.6f} s, max={all_thr_deltas.max():.6f} s")
    print(f"Cross-correlation lag over all stations:           mean={all_xcorr_lags.mean():.6f} s, median={np.median(all_xcorr_lags):.6f} s, min={all_xcorr_lags.min():.6f} s, max={all_xcorr_lags.max():.6f} s")


def plot_comparison(uobs_time, uobs_disp, sim_time, sim_disp, 
                    stations, tmax=2.0, output='comparison.png'):
    """Plot Uobs vs simulated traces for selected stations."""
    components = ['Ux', 'Uy', 'Uz']
    n = len(stations)
    
    fig, axes = plt.subplots(n, 3, figsize=(16, 2.5 * n), sharex=True)
    if n == 1:
        axes = axes[np.newaxis, :]
    
    fig.suptitle('Comparison: Observed (Uobs) vs Simulated Displacements', 
                 fontsize=14, fontweight='bold')
    
    for i, station in enumerate(stations):
        for j, comp in enumerate(components):
            ax = axes[i, j]
            ax.plot(uobs_time, uobs_disp[station][:, j], 'b-', 
                    linewidth=0.6, label='Uobs', alpha=0.7)
            ax.plot(sim_time, sim_disp[station][:, j], 'r-', 
                    linewidth=0.8, label='Simulated', alpha=0.9)
            ax.set_xlim(0, tmax)
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.set_title(comp, fontsize=12)
            if j == 0:
                ax.set_ylabel(f'{station}\nDispl (m)', fontsize=9)
            if i == n - 1:
                ax.set_xlabel('Time (s)')
            if i == 0 and j == 2:
                ax.legend(loc='upper right', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches='tight')
    print(f"Saved: {output}")


def main():
    parser = argparse.ArgumentParser(description='Compare two trace files')
    parser.add_argument('--uobs', type=str, default='Uobs/capteurs.0025.h5')
    parser.add_argument('--sim', type=str, default='SEM3D_ST7/traces/')
    parser.add_argument('--stations', type=str, nargs='+', default=['0000', '0030', '0060'])
    parser.add_argument('--tmax', type=float, default=2.0)
    parser.add_argument('--output', type=str, default='comparison.png')
    parser.add_argument('--arrival-frac', type=float, default=0.05, help='Fraction of peak norm used for arrival comparison')
    args = parser.parse_args()
    
    # Load data
    print(f"Loading Uobs: {args.uobs}")
    uobs_time, uobs_disp, uobs_pos = load_traces(args.uobs)
    print(f"  {len(uobs_disp)} stations, {len(uobs_time)} time steps")
    
    print(f"Loading Sim:  {args.sim}")
    sim_time, sim_disp, sim_pos = load_traces(args.sim)
    print(f"  {len(sim_disp)} stations, {len(sim_time)} time steps")
    
    # Stats
    uobs_stats = amplitude_stats(uobs_disp)
    sim_stats = amplitude_stats(sim_disp)
    print_comparison(uobs_stats, sim_stats)
    
    # Arrival-time comparison
    station_names = [f'Uobs_{s}' for s in args.stations]
    station_names = [s for s in station_names if s in uobs_disp and s in sim_disp]
    print_arrival_comparison(uobs_time, uobs_disp, sim_time, sim_disp, station_names, fraction=args.arrival_frac)

    # Plot
    plot_comparison(uobs_time, uobs_disp, sim_time, sim_disp, station_names, tmax=args.tmax, output=args.output)


if __name__ == '__main__':
    main()
