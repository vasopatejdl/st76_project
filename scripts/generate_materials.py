#!/usr/bin/env python3.11
"""
Generate Lambda/Mu/Density HDF5 material files for SEM3D.

Usage:
    python3.11 scripts/generate_materials.py                        
    python3.11 scripts/generate_materials.py --scale 49 --nu 0.3
    python3.11 scripts/generate_materials.py --scale 49 --nz 201     # finer Z sampling
"""

import os
import h5py
import numpy as np
import argparse


def generate_linear_gradient(xmin, xmax, ymin, ymax, zmin, zmax,
                             nx, ny, nz, scale, nu, rho):
    """Generate linear gradient Lamé parameters.
    
    Lambda = (100 + 0.45 * |z|) * 1e6 * scale
    Mu = 0.5 * (1 - 2*nu) * Lambda / nu
    """
    xv = np.linspace(xmin, xmax, nx, dtype=np.float64)
    yv = np.linspace(ymin, ymax, ny, dtype=np.float64)
    zv = np.linspace(zmin, zmax, nz, dtype=np.float64)
    xg, yg, zg = np.meshgrid(xv, yv, zv, indexing='xy')
    
    la = (100.0 + 0.45 * np.abs(zg)) * 1.e6 * scale
    mu = 0.5 * (1.0 - 2.0 * nu) * la / nu
    ds = np.full_like(zg, rho)
    
    return la, mu, ds


def compute_travel_time(scale, nu=0.3, rho=2000.0, z_source=-1000):
    """Compute P-wave travel time from z_source to surface."""
    z = np.linspace(z_source, 0, 10000)
    dz = np.abs(z[1] - z[0])
    
    la = (100.0 + 0.45 * np.abs(z)) * 1.e6 * scale
    mu = 0.5 * (1.0 - 2.0 * nu) * la / nu
    vp = np.sqrt((la + 2.0 * mu) / rho)
    
    return np.sum(dz / vp)


def write_material_h5(filepath, data, lims):
    """Write material property to HDF5 in SEM3D format."""
    trnsp = (2, 1, 0)  # SEM3D expects z,y,x ordering
    with h5py.File(filepath, 'w') as f:
        f.create_dataset('samples', data=data.transpose(*trnsp))
        f.attrs['xMinGlob'] = np.array([lims[0], lims[2], lims[4]])
        f.attrs['xMaxGlob'] = np.array([lims[1], lims[3], lims[5]])


def main():
    parser = argparse.ArgumentParser(description='Generate SEM3D material files')
    parser.add_argument('--scale', type=float, default=49.0, help='Scale factor for Lambda')
    parser.add_argument('--nu', type=float, default=0.3)
    parser.add_argument('--rho', type=float, default=2000.0, help='Density (kg/m^3)')
    parser.add_argument('--nx', type=int, default=49, help='Grid points in X')
    parser.add_argument('--ny', type=int, default=49, help='Grid points in Y')
    parser.add_argument('--nz', type=int, default=151, help='Grid points in Z')
    parser.add_argument('--xlim', type=float, nargs=2, default=[-1200, 1200])
    parser.add_argument('--ylim', type=float, nargs=2, default=[-1200, 1200])
    parser.add_argument('--zlim', type=float, nargs=2, default=[-1500, 5])
    parser.add_argument('--outdir', type=str, default='SEM3D_ST7')
    parser.add_argument('--prefix', type=str, default='example')
    args = parser.parse_args()
    
    xmin, xmax = args.xlim
    ymin, ymax = args.ylim
    zmin, zmax = args.zlim
    
    print(f"Scale: {args.scale}, nu: {args.nu}, rho: {args.rho}")
    print(f"Grid: {args.nx} x {args.ny} x {args.nz}")
    print(f"Domain: X=[{xmin},{xmax}] Y=[{ymin},{ymax}] Z=[{zmin},{zmax}]")
    print(f"Step sizes: X={abs(xmax-xmin)/(args.nx-1):.0f}m, "
          f"Y={abs(ymax-ymin)/(args.ny-1):.0f}m, "
          f"Z={abs(zmax-zmin)/(args.nz-1):.1f}m")
    
    # Generate
    la, mu, ds = generate_linear_gradient(
        xmin, xmax, ymin, ymax, zmin, zmax,
        args.nx, args.ny, args.nz, args.scale, args.nu, args.rho)
    
    vp = np.sqrt((la + 2.0 * mu) / ds)
    vs = np.sqrt(mu / ds)
    
    print(f"\nLambda: [{la.min()/1e6:.0f}, {la.max()/1e6:.0f}] MPa")
    print(f"Mu:     [{mu.min()/1e6:.0f}, {mu.max()/1e6:.0f}] MPa")
    print(f"Vp:     [{vp.min():.0f}, {vp.max():.0f}] m/s")
    print(f"Vs:     [{vs.min():.0f}, {vs.max():.0f}] m/s")
    
    # Travel time estimate
    tt = compute_travel_time(args.scale, args.nu, args.rho)
    print(f"\nP-wave travel time (z=-1000m to surface): {tt:.3f} s")
    
    # Write
    lims = (xmin, xmax, ymin, ymax, zmin, zmax)
    for name, data in [('la', la), ('mu', mu), ('ds', ds)]:
        fn = os.path.join(args.outdir, f'{args.prefix}_{name}.h5')
        write_material_h5(fn, data, lims)
        print(f"Written: {fn}")


if __name__ == '__main__':
    main()
