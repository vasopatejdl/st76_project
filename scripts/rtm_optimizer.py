#!/usr/bin/env python3.11
"""
RTM Optimizer using L-BFGS with backtracking line search.

Usage:
    # Initialize the state if it0
    1. python3.11 scripts/rtm_optimizer.py init 
    # Run these
    2. python3.11 scripts/rtm_optimizer.py gradient
    3. python3.11 scripts/rtm_optimizer.py propose
    4 .python3.11 scripts/rtm_optimizer.py check --from_traces 
    #If step 4 says Rejected go back to step 2, if acctepted run adjoint with new La, Mu.
"""

import h5py
import numpy as np
import os
import json
import argparse
import sys

sys.path.insert(0, os.path.dirname(__file__))
from compare_traces import load_traces


def load_material_h5(filepath):
    with h5py.File(filepath, 'r') as f:
        return f['samples'][...].copy(), f.attrs['xMinGlob'].copy(), f.attrs['xMaxGlob'].copy()


def save_material_h5(filepath, data, xmin, xmax):
    data = np.asarray(data, dtype=np.float64)
    xmin = np.asarray(xmin)
    xmax = np.asarray(xmax)
    with h5py.File(filepath, 'w') as f:
        f.create_dataset('samples', data=data)
        f.attrs['xMinGlob'] = xmin
        f.attrs['xMaxGlob'] = xmax


def load_gradients_on_grid(grad_dir, la_xmin, la_xmax, grid_shape):
    """Load node-level gradients and bin onto material grid via 3D nearest-neighbor averaging."""
    grad_file = os.path.join(grad_dir, 'gradients.h5')
    with h5py.File(grad_file, 'r') as f:
        gl = f['g_lambda'][...]
        gm = f['g_mu'][...]
        nodes = f['Nodes'][...]

    nz, ny, nx = grid_shape
    dx = (la_xmax[0] - la_xmin[0]) / (nx - 1)
    dy = (la_xmax[1] - la_xmin[1]) / (ny - 1)
    dz = (la_xmax[2] - la_xmin[2]) / (nz - 1)

    ix = np.clip(np.round((nodes[:, 0] - la_xmin[0]) / dx).astype(int), 0, nx - 1)
    iy = np.clip(np.round((nodes[:, 1] - la_xmin[1]) / dy).astype(int), 0, ny - 1)
    iz = np.clip(np.round((nodes[:, 2] - la_xmin[2]) / dz).astype(int), 0, nz - 1)

    gl_3d = np.zeros((nz, ny, nx), dtype=np.float64)
    gm_3d = np.zeros((nz, ny, nx), dtype=np.float64)
    counts = np.zeros((nz, ny, nx), dtype=np.float64)

    np.add.at(gl_3d, (iz, iy, ix), gl)
    np.add.at(gm_3d, (iz, iy, ix), gm)
    np.add.at(counts, (iz, iy, ix), 1.0)

    mask = counts > 0
    gl_3d[mask] /= counts[mask]
    gm_3d[mask] /= counts[mask]

    return gl_3d, gm_3d


def grid_spacing(xmin, xmax, shape):
    """Return (dx, dy, dz) for a regular grid stored as (nz, ny, nx)."""
    nz, ny, nx = shape
    dx = (xmax[0] - xmin[0]) / max(nx - 1, 1)
    dy = (xmax[1] - xmin[1]) / max(ny - 1, 1)
    dz = (xmax[2] - xmin[2]) / max(nz - 1, 1)
    return float(dx), float(dy), float(dz)


def discrete_laplacian(field, dx, dy, dz):
    """Second-order finite-difference Laplacian with edge replication."""
    lap = np.zeros_like(field, dtype=np.float64)

    if field.shape[2] > 2:
        lap[:, :, 1:-1] += (field[:, :, 2:] - 2.0 * field[:, :, 1:-1] + field[:, :, :-2]) / dx**2
        lap[:, :, 0] = lap[:, :, 1]
        lap[:, :, -1] = lap[:, :, -2]
    if field.shape[1] > 2:
        lap[:, 1:-1, :] += (field[:, 2:, :] - 2.0 * field[:, 1:-1, :] + field[:, :-2, :]) / dy**2
        lap[:, 0, :] = lap[:, 1, :]
        lap[:, -1, :] = lap[:, -2, :]
    if field.shape[0] > 2:
        lap[1:-1, :, :] += (field[2:, :, :] - 2.0 * field[1:-1, :, :] + field[:-2, :, :]) / dz**2
        lap[0, :, :] = lap[1, :, :]
        lap[-1, :, :] = lap[-2, :, :]

    return lap


def smoothness_regularization_value(field, xmin, xmax, weight):
    """Compute 0.5 * weight * integral |grad(field)|^2 dv on the material grid."""
    if weight == 0.0:
        return 0.0
    dx, dy, dz = grid_spacing(xmin, xmax, field.shape)
    gz, gy, gx = np.gradient(field, dz, dy, dx, edge_order=1)
    cell_vol = dx * dy * dz
    return 0.5 * weight * float(np.sum(gx**2 + gy**2 + gz**2)) * cell_vol


def smoothness_regularization_gradient(field, xmin, xmax, weight):
    """Gradient of 0.5 * weight * integral |grad(field)|^2 dv: -weight * Laplacian(field)."""
    if weight == 0.0:
        return np.zeros_like(field)
    dx, dy, dz = grid_spacing(xmin, xmax, field.shape)
    return -weight * discrete_laplacian(field, dx, dy, dz)


def compute_misfit(uobs_path, sim_path, la_file=None, mu_file=None, R_lambda=0.0, R_mu=0.0):
    """Compute full objective: data misfit + smoothness regularization."""
    from scipy.interpolate import interp1d as interp1d_

    uobs_time, uobs_disp, _ = load_traces(uobs_path)
    sim_time, sim_disp, _ = load_traces(sim_path)
    stations = sorted(set(uobs_disp.keys()) & set(sim_disp.keys()))
    dt = uobs_time[1] - uobs_time[0]

    J_data = 0.0
    for s in stations:
        sim_interp = np.zeros_like(uobs_disp[s])
        for c in range(3):
            f = interp1d_(sim_time, sim_disp[s][:, c], kind='linear',
                         fill_value=0.0, bounds_error=False)
            sim_interp[:, c] = f(uobs_time)
        J_data += 0.5 * np.sum((sim_interp - uobs_disp[s])**2) * dt

    J_reg = 0.0
    if (R_lambda != 0.0 or R_mu != 0.0) and la_file is not None and mu_file is not None:
        la_data, la_xmin, la_xmax = load_material_h5(la_file)
        mu_data, mu_xmin, mu_xmax = load_material_h5(mu_file)
        J_reg += smoothness_regularization_value(la_data, la_xmin, la_xmax, R_lambda)
        J_reg += smoothness_regularization_value(mu_data, mu_xmin, mu_xmax, R_mu)

    return J_data + J_reg


def add_regularization(g_la, g_mu, la_data, mu_data,
                       la_xmin, la_xmax, mu_xmin, mu_xmax,
                       R_la, R_mu):
    """Add PDF-style smoothness regularization: g_total = g_mis - R * Laplacian(param)."""
    g_la += smoothness_regularization_gradient(la_data, la_xmin, la_xmax, R_la)
    g_mu += smoothness_regularization_gradient(mu_data, mu_xmin, mu_xmax, R_mu)
    return g_la, g_mu


def lbfgs_direction(g, history, m=5):
    """L-BFGS two-loop recursion. Returns descent direction."""
    q = g.copy()

    hist = history[-m:] if len(history) > m else history
    k = len(hist)

    if k == 0:
        return -g

    alphas = np.zeros(k)
    rhos = np.zeros(k)

    for i in range(k):
        s_i, y_i = hist[i]
        sy = np.dot(s_i, y_i)
        rhos[i] = 1.0 / sy if abs(sy) > 1e-30 else 0.0

    for i in range(k - 1, -1, -1):
        s_i, y_i = hist[i]
        alphas[i] = rhos[i] * np.dot(s_i, q)
        q = q - alphas[i] * y_i

    s_last, y_last = hist[-1]
    yy = np.dot(y_last, y_last)
    gamma = np.dot(s_last, y_last) / yy if abs(yy) > 1e-30 else 1.0
    r = gamma * q

    for i in range(k):
        s_i, y_i = hist[i]
        beta = rhos[i] * np.dot(y_i, r)
        r = r + s_i * (alphas[i] - beta)

    return -r


def load_state(state_dir):
    state_file = os.path.join(state_dir, 'state.json')
    history_file = os.path.join(state_dir, 'lbfgs_history.npz')

    with open(state_file) as f:
        state = json.load(f)

    # Backward compatibility with older single-alpha state files.
    if 'alpha_lambda' not in state:
        alpha = state.get('alpha', 1.0)
        state['alpha_lambda'] = alpha
        state['alpha_mu'] = alpha
    if 'dir_deriv_lambda' not in state:
        total = state.get('dir_deriv', 0.0)
        state['dir_deriv_lambda'] = 0.0
        state['dir_deriv_mu'] = total
        state['dir_deriv'] = total
    if 'R_lambda_active' not in state:
        state['R_lambda_active'] = None
    if 'R_mu_active' not in state:
        state['R_mu_active'] = None

    history = []
    if os.path.exists(history_file):
        data = np.load(history_file, allow_pickle=True)
        history = list(data['history'])

    return state, history


def save_state(state_dir, state, history):
    os.makedirs(state_dir, exist_ok=True)
    with open(os.path.join(state_dir, 'state.json'), 'w') as f:
        json.dump(state, f, indent=2)
    np.savez(os.path.join(state_dir, 'lbfgs_history.npz'),
             history=np.array(history, dtype=object))


def cmd_gradient(args):
    """Load gradients, add regularization, compute L-BFGS search directions."""
    state, history = load_state(args.state_dir)

    la_data, la_xmin, la_xmax = load_material_h5(args.la_file)
    mu_data, mu_xmin, mu_xmax = load_material_h5(args.mu_file)

    g_la_mis, g_mu_mis = load_gradients_on_grid(
        args.grad_dir, la_xmin, la_xmax, la_data.shape)

    print(f"Misfit gradient: g_la max={np.abs(g_la_mis).max()}, g_mu max={np.abs(g_mu_mis).max()}")

    R_la = args.R_lambda
    R_mu = args.R_mu
    if args.adaptive_regularization:
        g_la_reg_unit = smoothness_regularization_gradient(la_data, la_xmin, la_xmax, 1.0)
        g_mu_reg_unit = smoothness_regularization_gradient(mu_data, mu_xmin, mu_xmax, 1.0)

        la_mis_norm = np.linalg.norm(g_la_mis.ravel())
        mu_mis_norm = np.linalg.norm(g_mu_mis.ravel())
        la_reg_norm = np.linalg.norm(g_la_reg_unit.ravel())
        mu_reg_norm = np.linalg.norm(g_mu_reg_unit.ravel())

        R_la = args.wp_lambda * la_mis_norm / la_reg_norm if la_reg_norm > 0 else 0.0
        R_mu = args.wp_mu * mu_mis_norm / mu_reg_norm if mu_reg_norm > 0 else 0.0

        print("Adaptive regularization (Fathi et al. 2015):")
        print(f"  ||g_la_mis||={la_mis_norm}, ||g_la_reg||={la_reg_norm} -> R_lambda={R_la} (wp={args.wp_lambda})")
        print(f"  ||g_mu_mis||={mu_mis_norm}, ||g_mu_reg||={mu_reg_norm} -> R_mu={R_mu} (wp={args.wp_mu})")

    g_la = g_la_mis.copy()
    g_mu = g_mu_mis.copy()
    g_la, g_mu = add_regularization(
        g_la, g_mu, la_data, mu_data,
        la_xmin, la_xmax, mu_xmin, mu_xmax,
        R_la, R_mu)

    print(f"Total gradient:  g_la max={np.abs(g_la).max():}, g_mu max={np.abs(g_mu).max():}")

    J_current = compute_misfit(args.uobs, args.sim,
                               la_file=args.la_file, mu_file=args.mu_file,
                               R_lambda=R_la, R_mu=R_mu)
    if state.get('J_history'):
        state['J_history'][-1] = float(J_current)
    else:
        state['J_history'] = [float(J_current)]
    print(f"Current objective with active regularization: J = {J_current:}")

    g_flat = np.concatenate([g_la.ravel(), g_mu.ravel()])

    pending_file = os.path.join(args.state_dir, 'pending_s.npz')
    prev_g_file = os.path.join(args.state_dir, 'prev_g.npz')
    if os.path.exists(pending_file) and os.path.exists(prev_g_file):
        s_flat = np.asarray(np.load(pending_file, allow_pickle=True)['s_flat'], dtype=np.float64)
        g_prev = np.asarray(np.load(prev_g_file, allow_pickle=True)['g_flat'], dtype=np.float64)
        y_flat = g_flat - g_prev

        sy = np.dot(s_flat, y_flat)
        if sy > 1e-30:
            history.append((s_flat, y_flat))
            print(f"L-BFGS: added history pair (now {len(history)} pairs, s·y={sy:})")
        else:
            print(f"L-BFGS: skipped pair (s·y={sy:} <= 0, curvature violated)")
        os.remove(pending_file)

    np.savez(prev_g_file, g_flat=g_flat)

    d_flat = lbfgs_direction(g_flat, history, m=args.lbfgs_m)

    n = g_la.size
    d_la = d_flat[:n].reshape(la_data.shape)
    d_mu = d_flat[n:].reshape(mu_data.shape)

    dir_deriv_la = float(np.dot(g_la.ravel(), d_la.ravel()))
    dir_deriv_mu = float(np.dot(g_mu.ravel(), d_mu.ravel()))
    dir_deriv = dir_deriv_la + dir_deriv_mu
    print(f"Directional derivative: {dir_deriv} {'✓ descent' if dir_deriv < 0 else '✗ NOT descent'}")
    print(f"  lambda contribution: {dir_deriv_la}")
    print(f"  mu contribution:     {dir_deriv_mu}")

    if dir_deriv >= 0:
        print("  Resetting to steepest descent")
        d_la = -g_la
        d_mu = -g_mu
        d_flat = np.concatenate([d_la.ravel(), d_mu.ravel()])
        dir_deriv_la = float(np.dot(g_la.ravel(), d_la.ravel()))
        dir_deriv_mu = float(np.dot(g_mu.ravel(), d_mu.ravel()))
        dir_deriv = dir_deriv_la + dir_deriv_mu
        history.clear()

    alpha_la = 1.0
    alpha_mu = 1.0

    np.savez(os.path.join(args.state_dir, 'current_search.npz'),
             d_la=d_la, d_mu=d_mu, g_la=g_la, g_mu=g_mu,
             g_flat=g_flat, d_flat=d_flat,
             la_data=la_data, mu_data=mu_data,
             la_xmin=la_xmin, la_xmax=la_xmax,
             mu_xmin=mu_xmin, mu_xmax=mu_xmax)

    state['alpha_lambda'] = float(alpha_la)
    state['alpha_mu'] = float(alpha_mu)
    state['R_lambda_active'] = float(R_la)
    state['R_mu_active'] = float(R_mu)
    state['dir_deriv_lambda'] = float(dir_deriv_la)
    state['dir_deriv_mu'] = float(dir_deriv_mu)
    state['dir_deriv'] = float(dir_deriv)
    state['accepted'] = False
    save_state(args.state_dir, state, history)

    print(f"\nIteration {state['iteration']}: initial alpha_lambda = {alpha_la}, alpha_mu = {alpha_mu}")


def cmd_propose(args):
    """Write trial materials with current alpha_lambda and alpha_mu."""
    state, history = load_state(args.state_dir)
    search = np.load(os.path.join(args.state_dir, 'current_search.npz'), allow_pickle=True)

    alpha_la = state['alpha_lambda']
    alpha_mu = state['alpha_mu']
    la_data = np.asarray(search['la_data'], dtype=np.float64)
    mu_data = np.asarray(search['mu_data'], dtype=np.float64)
    d_la = np.asarray(search['d_la'], dtype=np.float64)
    d_mu = np.asarray(search['d_mu'], dtype=np.float64)

    la_new = np.maximum(la_data + alpha_la * d_la, 0.01 * la_data.min())
    mu_new = np.maximum(mu_data + alpha_mu * d_mu, 0.01 * mu_data.min())

    save_material_h5(args.la_file, la_new, search['la_xmin'], search['la_xmax'])
    save_material_h5(args.mu_file, mu_new, search['mu_xmin'], search['mu_xmax'])

    print(f"Trial update with alpha_lambda = {alpha_la}, alpha_mu = {alpha_mu}:")
    print(f"  Lambda: max change = {(np.abs(la_new - la_data) / (np.abs(la_data) + 1e-30)).max():}")
    print(f"  Mu:     max change = {(np.abs(mu_new - mu_data) / (np.abs(mu_data) + 1e-30)).max():}")


def cmd_check(args):
    """Check Armijo condition and accept or reduce alpha_lambda, alpha_mu."""
    state, history = load_state(args.state_dir)
    search = np.load(os.path.join(args.state_dir, 'current_search.npz'), allow_pickle=True)

    R_la = state.get('R_lambda_active')
    R_mu = state.get('R_mu_active')
    if R_la is None:
        R_la = args.R_lambda
    if R_mu is None:
        R_mu = args.R_mu

    if args.from_traces:
        J_trial = compute_misfit(args.uobs, args.sim,
                                 la_file=args.la_file, mu_file=args.mu_file,
                                 R_lambda=R_la, R_mu=R_mu)
        print(f"Computed J_trial = {J_trial}")
    else:
        J_trial = args.J_trial
        print(f"Provided J_trial = {J_trial}")

    J_current = state['J_history'][-1]
    alpha_la = state['alpha_lambda']
    alpha_mu = state['alpha_mu']
    dir_deriv_la = state['dir_deriv_lambda']
    dir_deriv_mu = state['dir_deriv_mu']
    armijo_rhs = J_current + args.c1 * (alpha_la * dir_deriv_la + alpha_mu * dir_deriv_mu)

    print(f"\nArmijo check:")
    print(f"  J_current = {J_current:}")
    print(f"  J_trial   = {J_trial:}")
    print(f"  RHS       = {armijo_rhs:}")
    print(f"  alpha_lambda = {alpha_la:}")
    print(f"  alpha_mu     = {alpha_mu:}")
    print(f"  Decrease  = {J_current - J_trial:} ({(J_current - J_trial) / J_current * 100}%)")

    if J_trial < armijo_rhs:
        print(f"\n  ✓ ACCEPTED (alpha_lambda = {alpha_la:}, alpha_mu = {alpha_mu:})")

        s_flat = np.concatenate([
            (alpha_la * np.asarray(search['d_la'], dtype=np.float64)).ravel(),
            (alpha_mu * np.asarray(search['d_mu'], dtype=np.float64)).ravel(),
        ]).astype(np.float64, copy=False)
        np.savez(os.path.join(args.state_dir, 'pending_s.npz'), s_flat=s_flat)

        state['J_history'].append(float(J_trial))
        state['iteration'] += 1
        state['accepted'] = True
        state['alpha_lambda'] = 1.0
        state['alpha_mu'] = 1.0
        save_state(args.state_dir, state, history)
    else:
        new_alpha_la = alpha_la * args.xi
        new_alpha_mu = alpha_mu * args.xi
        print(f"\n  ✗ REJECTED — reducing alpha_lambda: {alpha_la:} → {new_alpha_la:}")
        print(f"                 reducing alpha_mu:     {alpha_mu:} → {new_alpha_mu:}")

        save_material_h5(args.la_file, search['la_data'],
                        search['la_xmin'], search['la_xmax'])
        save_material_h5(args.mu_file, search['mu_data'],
                        search['mu_xmin'], search['mu_xmax'])

        state['alpha_lambda'] = float(new_alpha_la)
        state['alpha_mu'] = float(new_alpha_mu)
        save_state(args.state_dir, state, history)

        print(f"  Materials reverted.")


def cmd_status(args):
    """Print optimizer status."""
    state, history = load_state(args.state_dir)

    print(f"Iteration: {state['iteration']}")
    print(f"Alpha_lambda: {state.get('alpha_lambda', state.get('alpha', 1.0))}")
    print(f"Alpha_mu:     {state.get('alpha_mu', state.get('alpha', 1.0))}")
    print(f"Accepted: {state['accepted']}")
    print(f"L-BFGS history pairs: {len(history)}")

    if state['J_history']:
        print(f"\nMisfit history:")
        for i, J in enumerate(state['J_history']):
            reduction = (state['J_history'][0] - J) / state['J_history'][0] * 100 if i > 0 else 0
            print(f"  Iter {i}: J = {J} ({reduction:+}%)")


def cmd_init(args):
    """Initialize optimizer with baseline misfit."""
    J0 = compute_misfit(args.uobs, args.sim,
                        la_file=args.la_file, mu_file=args.mu_file,
                        R_lambda=args.R_lambda, R_mu=args.R_mu)
    print(f"Baseline misfit J0 = {J0}")

    state = {
        'iteration': 0,
        'alpha_lambda': 1.0,
        'alpha_mu': 1.0,
        'R_lambda_active': float(args.R_lambda),
        'R_mu_active': float(args.R_mu),
        'J_history': [float(J0)],
        'accepted': True,
        'dir_deriv_lambda': 0.0,
        'dir_deriv_mu': 0.0,
        'dir_deriv': 0.0,
    }
    os.makedirs(args.state_dir, exist_ok=True)
    save_state(args.state_dir, state, [])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['init', 'gradient', 'propose', 'check', 'status'])
    parser.add_argument('--la_file', default='SEM3D_ST7/example_la.h5')
    parser.add_argument('--mu_file', default='SEM3D_ST7/example_mu.h5')
    parser.add_argument('--grad_dir', default='SEM3D_ST7/gradients')
    parser.add_argument('--uobs', default='Uobs/capteurs.0025.h5')
    parser.add_argument('--sim', default='SEM3D_ST7/traces/')
    parser.add_argument('--state_dir', default='SEM3D_ST7/optim_state')
    parser.add_argument('--R_lambda', type=float, default=1e-4)
    parser.add_argument('--R_mu', type=float, default=1e-4)
    parser.add_argument('--adaptive-regularization', action='store_true',
        help='Choose R_lambda and R_mu adaptively from gradient norms per Fathi et al. (2015)')
    parser.add_argument('--wp_lambda', type=float, default=0.5)
    parser.add_argument('--wp_mu', type=float, default=0.5)
    parser.add_argument('--lbfgs_m', type=int, default=15)
    parser.add_argument('--c1', type=float, default=1e-4)
    parser.add_argument('--xi', type=float, default=0.5)
    parser.add_argument('--J_trial', type=float, default=None)
    parser.add_argument('--from_traces', action='store_true')
    args = parser.parse_args()

    cmds = {
        'init': cmd_init,
        'gradient': cmd_gradient,
        'propose': cmd_propose,
        'check': cmd_check,
        'status': cmd_status,
    }
    cmds[args.command](args)


if __name__ == '__main__':
    main()
