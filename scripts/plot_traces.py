#!/usr/bin/env python3.11
"""
Plot traces from a single SEM3D HDF5 capteurs file.

Usage:
    python3.11 scripts/plot_traces.py Uobs/capteurs.0025.h5
    python3.11 scripts/plot_traces.py SEM3D_ST7/traces/capteurs.0065.h5 --stations 0000 0060 0120
    python3.11 scripts/plot_traces.py Uobs/capteurs.0025.h5 --seismogram --nstations 50
"""

import argparse

import h5py
import matplotlib.pyplot as plt
import numpy as np


def load_traces(filepath, displ_cols=(2, 3, 4)):
    """Load displacement traces from one or more SEM3D HDF5 capteurs files.
    
    filepath can be a single .h5 file or a directory containing capteurs.*.h5 files.
    """
    import glob
    import os
    
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


def plot_timeseries(time, disp, stations, tmax=None, output='traces.png', title=''):
    """Plot time series for selected stations."""
    components = ['Ux', 'Uy', 'Uz']
    n = len(stations)
    colors = plt.cm.tab10(np.linspace(0, 1, n))
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(title or 'Displacement Time Histories', fontsize=14, fontweight='bold')
    
    for j, (ax, comp) in enumerate(zip(axes, components)):
        for i, station in enumerate(stations):
            ax.plot(time, disp[station][:, j], color=colors[i], 
                    linewidth=0.7, label=station, alpha=0.8)
        ax.set_ylabel(f'{comp} (m)')
        ax.grid(True, alpha=0.3)
        if j == 0:
            ax.legend(loc='upper right', fontsize=7, ncol=min(n, 5))
        if tmax:
            ax.set_xlim(0, tmax)
    
    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches='tight')
    print(f"Saved: {output}")


def plot_seismogram(time, disp, pos, nstations=50, tmax=None, output='seismogram.png', title=''):
    """Plot wiggle/seismogram view sorted by station X position."""
    components = ['Ux', 'Uy', 'Uz']
    station_names = sorted(disp.keys())
    
    # Sort by X position
    pos_arr = np.array([pos[k] for k in station_names])
    order = np.argsort(pos_arr[:, 0])
    sorted_names = [station_names[i] for i in order[:nstations]]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 10), sharey=True)
    fig.suptitle(title or 'Seismogram (Wiggle Plot)', fontsize=14, fontweight='bold')
    
    for j, (ax, comp) in enumerate(zip(axes, components)):
        for i, name in enumerate(sorted_names):
            u = disp[name][:, j]
            u_norm = u / (np.max(np.abs(u)) + 1e-30) * 0.4
            ax.plot(time, i + u_norm, 'k-', linewidth=0.5, alpha=0.7)
            ax.fill_between(time, i, i + u_norm, where=u_norm > 0,
                            facecolor='red', alpha=0.4)
            ax.fill_between(time, i, i + u_norm, where=u_norm < 0,
                            facecolor='blue', alpha=0.4)
        ax.set_xlabel('Time (s)')
        ax.set_title(comp)
        if j == 0:
            ax.set_ylabel('Station index (sorted by X)')
        if tmax:
            ax.set_xlim(0, tmax)
        ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches='tight')
    print(f"Saved: {output}")


def main():
    parser = argparse.ArgumentParser(description='Plot SEM3D traces')
    parser.add_argument('filepath', type=str, help='Path to HDF5 capteurs file')
    parser.add_argument('--stations', type=str, nargs='+', default=None)                        
    parser.add_argument('--tmax', type=float, default=None, help='Max time for x-axis')
    parser.add_argument('--output', type=str, default=None, help='Output filename')
    parser.add_argument('--seismogram', action='store_true', help='Plot seismogram/wiggle view')
    parser.add_argument('--nstations', type=int, default=50, help='Stations for seismogram view')
    parser.add_argument('--title', type=str, default='', help='Plot title')
    args = parser.parse_args()
    
    # Load
    print(f"Loading: {args.filepath}")
    time, disp, pos = load_traces(args.filepath)
    station_names = sorted(disp.keys())
    print(f"  {len(disp)} stations, {len(time)} time steps, "
          f"t=[{time[0]:.4f}, {time[-1]:.4f}] s")
    
    # Stats
    all_data = np.array(list(disp.values()))
    for i, c in enumerate(['Ux', 'Uy', 'Uz']):
        d = all_data[:, :, i]
        print(f"  {c}: max={np.abs(d).max():.2e}, rms={np.sqrt(np.mean(d**2)):.2e}")
    
    if args.seismogram:
        output = args.output or 'seismogram.png'
        plot_seismogram(time, disp, pos, nstations=args.nstations,
                        tmax=args.tmax, output=output, title=args.title)
    else:
        # Select stations
        if args.stations:
            selected = [f'Uobs_{s}' for s in args.stations]
        else:
            indices = np.linspace(0, len(station_names)-1, 5, dtype=int)
            selected = [station_names[i] for i in indices]
        selected = [s for s in selected if s in disp]
        
        output = args.output or 'traces.png'
        plot_timeseries(time, disp, selected, tmax=args.tmax,
                        output=output, title=args.title)


if __name__ == '__main__':
    main()
