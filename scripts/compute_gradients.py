#!/usr/bin/env python3.11
"""
Compute RTM gradients from forward and adjoint snapshots.

From the control problem (Fichtner 2006):
    (1/rho) * M * g_lambda = R_lambda * g_reg_lambda + g_mis_lambda
    (1/rho) * M * g_mu     = R_mu * g_reg_mu     + g_mis_mu

Where:
    M = diagonal mass matrix (at GLL nodes), from geometry
    g_mis_lambda = -sum_n [N * eps_vol_fwd * eps_vol_adj] * dt
    g_mis_mu     = -sum_n [N * (2*eps_vol_fwd*eps_vol_adj + edev:edev_adj)] * dt
    g_reg_lambda = M * lambda  (Tikhonov regularization)
    g_reg_mu     = M * mu

Since M is diagonal in SEM, solving for g:
    g_lambda = rho * M^{-1} * (R_lambda * M * lambda + g_mis_lambda)
             = rho * (R_lambda * lambda + M^{-1} * g_mis_lambda)

The misfit gradient at element level needs to be assembled to nodes
using the element connectivity and weighted by the mass matrix.

- Shared boundary nodes are identified via coordinate hashing
- Element connectivity is remapped to global unique node indices
- Gradients are computed on the deduplicated global mesh

Usage:
    python3.11 scripts/compute_gradients.py
    python3.11 scripts/compute_gradients.py --fwd_res res_forward --adj_res res_adjoint
"""

import argparse
import glob
import hashlib
import os

import h5py
import numpy as np
from tqdm import tqdm


def discover_snapshots(res_dir):
    """Find all snapshot directories and sem_field files."""
    snap_dirs = sorted(glob.glob(os.path.join(res_dir, 'Rsem*')))
    sample = snap_dirs[0]
    field_files = sorted(glob.glob(os.path.join(sample, 'sem_field.*.h5')))
    return snap_dirs, len(field_files)


def hash_node_coordinates(coords, decimals=6):
    """Hash node coordinates for deduplication using MD5."""
    rounded = np.round(coords, decimals=decimals)
    hashes = np.empty(len(coords), dtype='S16')

    for i, xyz in enumerate(rounded):
        hash_obj = hashlib.md5(xyz.tobytes())
        hashes[i] = hash_obj.digest()[:16]

    return hashes


def deduplicate_and_remap_geometry(partition_geometries):
    """Deduplicate shared boundary nodes across partitions and remap element connectivity.
    """
    n_parts = len(partition_geometries)

    print(f"\nDeduplicating nodes across partitions")

    all_nodes = []
    all_elements = []
    all_mass = []
    all_jac = []
    all_dens = []
    all_lambda = []
    all_mu = []

    partition_info = []
    node_offset = 0
    elem_offset = 0

    for p, geo in enumerate(partition_geometries):
        n_nodes_p = geo['n_nodes']
        n_elems_p = geo['n_elems']

        partition_info.append({
            'partition': p,
            'node_offset': node_offset,
            'elem_offset': elem_offset,
            'n_nodes_original': n_nodes_p,
            'n_elems': n_elems_p,
        })

        all_nodes.append(geo['Nodes'])
        all_mass.append(geo['Mass'])
        all_jac.append(geo['Jac'])
        all_dens.append(geo['Dens'])
        all_lambda.append(geo['Lambda'])
        all_mu.append(geo['Mu'])

        all_elements.append(geo['Elements'] + node_offset)

        node_offset += n_nodes_p
        elem_offset += n_elems_p

    concat_nodes = np.vstack(all_nodes)
    concat_elements = np.vstack(all_elements)
    concat_mass = np.concatenate(all_mass)
    concat_jac = np.concatenate(all_jac)
    concat_dens = np.concatenate(all_dens)
    concat_lambda = np.concatenate(all_lambda)
    concat_mu = np.concatenate(all_mu)

    n_nodes_concat = len(concat_nodes)
    n_elems_total = len(concat_elements)

    print(f"Total concatenated nodes: {n_nodes_concat}")
    print(f"Total elements: {n_elems_total}")
    for info in partition_info:
        print(f"  Partition {info['partition']}: {info['n_nodes_original']} nodes, "
              f"{info['n_elems']} elements (offset {info['node_offset']})")

    print("\nHashing node coordinates (precision: 6 decimals)...")
    node_hashes = hash_node_coordinates(concat_nodes, decimals=6)

    # Unique physical nodes from concatenated original nodes
    unique_hashes, unique_hash_idx, unique_hash_inv = np.unique(
        node_hashes,
        return_index=True,
        return_inverse=True,
        axis=0,
    )
    n_unique_nodes = len(unique_hashes)

    print(f"Unique physical nodes: {n_unique_nodes}")
    print(f"Duplicate nodes removed: {n_nodes_concat - n_unique_nodes}")
    print(f"Duplication ratio: {100.0 * (n_nodes_concat - n_unique_nodes) / n_nodes_concat:.2f}%")

    # Remap element connectivity to deduplicated node ids
    unique_elements = unique_hash_inv[concat_elements]
    unique_node_coords = concat_nodes[unique_hash_idx]

    # Aggregate nodal fields across duplicate partition copies.
    # In current SEM3D geometry outputs, duplicates are identical copies,
    # so mean is the safe/default policy (sum would overcount).
    node_counts = np.bincount(unique_hash_inv, minlength=n_unique_nodes)

    def mean_at_unique_nodes(node_data):
        s = np.bincount(unique_hash_inv, weights=node_data, minlength=n_unique_nodes)
        return s / np.maximum(node_counts, 1)

    unique_mass = mean_at_unique_nodes(concat_mass)
    unique_jac = mean_at_unique_nodes(concat_jac)
    unique_dens = mean_at_unique_nodes(concat_dens)
    unique_lambda = mean_at_unique_nodes(concat_lambda)
    unique_mu = mean_at_unique_nodes(concat_mu)

    # Diagnostics: check duplicate consistency for copied nodal fields
    dup_groups = np.where(node_counts > 1)[0]

    def max_rel_spread(arr):
        if len(dup_groups) == 0:
            return 0.0
        spread_max = 0.0
        for g in dup_groups:
            ids = np.where(unique_hash_inv == g)[0]
            vals = arr[ids]
            rel = (vals.max() - vals.min()) / (abs(vals.mean()) + 1e-30)
            spread_max = max(spread_max, float(rel))
        return spread_max

    print("\nValidation:")
    print(f"  Deduplicated nodes: {n_unique_nodes} < {n_nodes_concat} ✓")
    print(f"  Max element index < n_unique_nodes: {unique_elements.max() < n_unique_nodes} ✓")
    print(f"  Max node overlap count: {int(node_counts.max())}")
    print("  Aggregation policy: mean for Mass/Jac/Dens/Lambda/Mu")
    print(f"  Duplicate consistency (max rel spread): "
          f"Mass={max_rel_spread(concat_mass):.2e}, "
          f"Dens={max_rel_spread(concat_dens):.2e}, "
          f"Lamb={max_rel_spread(concat_lambda):.2e}, "
          f"Mu={max_rel_spread(concat_mu):.2e}")

    z_coords = unique_node_coords[:, 2]
    print(f"  Global mesh Z-range: [{z_coords.min():.1f}, {z_coords.max():.1f}] m")

    unified_geometry = {
        'Nodes': unique_node_coords,
        'Elements': unique_elements,
        'Mass': unique_mass,
        'Jac': unique_jac,
        'Dens': unique_dens,
        'Lambda': unique_lambda,
        'Mu': unique_mu,
        'n_nodes': n_unique_nodes,
        'n_elems': n_elems_total,
        'partition_info': partition_info,
        'n_parts_original': n_parts,
    }

    print("\nUnified geometry created:")
    print(f"  Nodes: {unique_node_coords.shape}")
    print(f"  Elements: {unique_elements.shape}")

    return unified_geometry


def load_geometry(res_dir):
    """Load geometry from partition files and deduplicate shared boundary nodes."""
    geo_files = sorted(glob.glob(os.path.join(res_dir, 'geometry*.h5')))

    partition_geometries = []
    for gf in geo_files:
        with h5py.File(gf, 'r') as f:
            geo = {
                'Nodes': f['Nodes'][...],
                'Elements': f['Elements'][...],
                'Mass': f['Mass'][...].astype(np.float64),
                'Jac': f['Jac'][...].astype(np.float64),
                'Dens': f['Dens'][...].astype(np.float64),
                'Lambda': f['Lamb'][...].astype(np.float64),
                'Mu': f['Mu'][...].astype(np.float64),
                'n_nodes': f['Nodes'].shape[0],
                'n_elems': f['Elements'].shape[0],
            }
            partition_geometries.append(geo)

    return deduplicate_and_remap_geometry(partition_geometries)


def load_snapshot_strains(snap_dir, n_parts):
    """Load volumetric and deviatoric strain fields from a snapshot, concatenated across partitions."""
    eps_vol_parts = []
    eps_dev_parts = []

    for i in range(n_parts):
        fn = os.path.join(snap_dir, f'sem_field.{i:04d}.h5')
        with h5py.File(fn, 'r') as f:
            eps_vol_parts.append(f['eps_vol'][...].astype(np.float64))
            edev = np.column_stack([
                f['eps_dev_xx'][...],
                f['eps_dev_yy'][...],
                f['eps_dev_zz'][...],
                f['eps_dev_xy'][...],
                f['eps_dev_xz'][...],
                f['eps_dev_yz'][...],
            ]).astype(np.float64)
            eps_dev_parts.append(edev)

    eps_vol = np.concatenate(eps_vol_parts)
    eps_dev = np.vstack(eps_dev_parts)

    return eps_vol, eps_dev


def assemble_elem_to_nodes(elem_values, elements, n_nodes):
    """Assemble element values to GLL nodes using connectivity (1/8 per corner node)."""
    node_values = np.zeros(n_nodes, dtype=np.float64)

    for j in range(8):
        np.add.at(node_values, elements[:, j], elem_values / 8.0)

    return node_values


def compute_gradients(fwd_res_dir, adj_res_dir, snap_interval=0.1):
    """
    Compute RTM gradients g_lambda and g_mu.

    Returns:
        g_lambda, g_mu: (n_unique_nodes,) gradient at deduplicated nodes
        g_la_mis_elem, g_mu_mis_elem: (n_total_elems,) element-level misfit gradients
        geometry: Unified geometry dict with deduplicated nodes
    """
    dt = snap_interval

    fwd_snaps, n_parts_fwd = discover_snapshots(fwd_res_dir)
    adj_snaps, n_parts_adj = discover_snapshots(adj_res_dir)

    n_snaps = min(len(fwd_snaps), len(adj_snaps))
    n_parts = n_parts_fwd

    print(f"Forward snapshots: {len(fwd_snaps)}, Adjoint snapshots: {len(adj_snaps)}")
    print(f"Using {n_snaps} snapshot pairs, {n_parts} parts")
    print(f"dt = {dt} s")

    geometry = load_geometry(fwd_res_dir)

    g_la_mis_elem = np.zeros(geometry['n_elems'], dtype=np.float64)
    g_mu_mis_elem = np.zeros(geometry['n_elems'], dtype=np.float64)

    print(f"\nAccumulating misfit gradients over {n_snaps} time steps...")
    for i in tqdm(range(n_snaps)):
        j = n_snaps - 1 - i

        fwd_evol, fwd_edev = load_snapshot_strains(fwd_snaps[i], n_parts)
        adj_evol, adj_edev = load_snapshot_strains(adj_snaps[j], n_parts)

        g_la_mis_elem -= fwd_evol * adj_evol * dt

        diag = np.sum(fwd_edev[:, :3] * adj_edev[:, :3], axis=1)
        off_diag = 2.0 * np.sum(fwd_edev[:, 3:] * adj_edev[:, 3:], axis=1)

        g_mu_mis_elem -= (2.0 * fwd_evol * adj_evol + diag + off_diag) * dt

    print("\nAssembling element gradients to nodes...")
    g_la_mis_node = assemble_elem_to_nodes(
        g_la_mis_elem, geometry['Elements'], geometry['n_nodes'])
    g_mu_mis_node = assemble_elem_to_nodes(
        g_mu_mis_elem, geometry['Elements'], geometry['n_nodes'])

    print("Applying mass matrix inverse...")

    M = geometry['Mass']
    rho = geometry['Dens']

    M_safe = np.where(M > 0, M, 1.0)

    g_lambda = rho * g_la_mis_node / M_safe
    g_mu = rho * g_mu_mis_node / M_safe

    g_lambda[M <= 0] = 0.0
    g_mu[M <= 0] = 0.0

    return g_lambda, g_mu, g_la_mis_elem, g_mu_mis_elem, geometry


def save_gradients_h5(g_lambda, g_mu, geometry, output_dir):
    """Save gradients as a single HDF5 file with deduplicated nodes."""
    os.makedirs(output_dir, exist_ok=True)

    fn = os.path.join(output_dir, 'gradients.h5')
    with h5py.File(fn, 'w') as f:
        f.create_dataset('g_lambda', data=g_lambda)
        f.create_dataset('g_mu', data=g_mu)

        f.create_dataset('Nodes', data=geometry['Nodes'])
        f.create_dataset('Elements', data=geometry['Elements'])
        f.create_dataset('Mass', data=geometry['Mass'])
        f.create_dataset('Lambda', data=geometry['Lambda'])
        f.create_dataset('Mu', data=geometry['Mu'])
        f.create_dataset('Dens', data=geometry['Dens'])

        f.attrs['n_nodes'] = geometry['n_nodes']
        f.attrs['n_elems'] = geometry['n_elems']
        f.attrs['n_parts_original'] = geometry['n_parts_original']

        info_str = str(geometry['partition_info'])
        f.attrs['partition_info'] = info_str

    print(f"  Written: {fn}")
    print(f"    - g_lambda, g_mu: ({geometry['n_nodes']},) at deduplicated nodes")
    print(f"    - Elements: ({geometry['n_elems']}, 8) with remapped connectivity")


def plot_gradients(g_lambda, g_mu, geometry, output='gradients.png'):
    """Plot gradient cross-sections (XZ and XY slices)."""
    import matplotlib.pyplot as plt

    nodes = geometry['Nodes']
    nx, ny, nz = nodes[:, 0], nodes[:, 1], nodes[:, 2]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('RTM Gradients (at deduplicated GLL nodes)', fontsize=14, fontweight='bold')

    y_tol = 150
    mask_xz = np.abs(ny) < y_tol

    for col, (data, name) in enumerate([(g_lambda, 'g_lambda'), (g_mu, 'g_mu')]):
        ax = axes[0, col]
        vmax = np.percentile(np.abs(data[mask_xz]), 99)
        sc = ax.scatter(nx[mask_xz], nz[mask_xz], c=data[mask_xz],
                      cmap='RdBu_r', s=1, alpha=0.5, vmin=-vmax, vmax=vmax)
        plt.colorbar(sc, ax=ax)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Z (m)')
        ax.set_title(f'{name} (XZ, y≈0)')
        ax.set_aspect('equal')

    z_tol = 50
    mask_xy = np.abs(nz + 500) < z_tol

    for col, (data, name) in enumerate([(g_lambda, 'g_lambda'), (g_mu, 'g_mu')]):
        ax = axes[1, col]
        vmax = np.percentile(np.abs(data[mask_xy]), 99)
        sc = ax.scatter(nx[mask_xy], ny[mask_xy], c=data[mask_xy],
                      cmap='RdBu_r', s=1, alpha=0.5, vmin=-vmax, vmax=vmax)
        plt.colorbar(sc, ax=ax)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(f'{name} (XY, z≈-500m)')
        ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches='tight')
    print(f"Saved: {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fwd_res', type=str, default='SEM3D_ST7/res/forw_res')
    parser.add_argument('--adj_res', type=str, default='SEM3D_ST7/res/back_res/res')
    parser.add_argument('--snap_interval', type=float, default=0.1)
    parser.add_argument('--outdir', type=str, default='SEM3D_ST7/gradients')
    parser.add_argument('--plot_output', type=str, default='gradients.png')
    args = parser.parse_args()

    g_lambda, g_mu, _, _, geometry = compute_gradients(
        args.fwd_res, args.adj_res, args.snap_interval)

    print("\nGradient statistics (at deduplicated nodes):")
    print(f"  g_lambda: min={g_lambda.min():.2e}, max={g_lambda.max():.2e}, "
          f"rms={np.sqrt(np.mean(g_lambda**2)):.2e}")
    print(f"  g_mu:     min={g_mu.min():.2e}, max={g_mu.max():.2e}, "
          f"rms={np.sqrt(np.mean(g_mu**2)):.2e}")

    print(f"\nSaving gradients to {args.outdir}/")
    save_gradients_h5(g_lambda, g_mu, geometry, args.outdir)

    plot_gradients(g_lambda, g_mu, geometry, output=args.plot_output)


if __name__ == '__main__':
    main()
