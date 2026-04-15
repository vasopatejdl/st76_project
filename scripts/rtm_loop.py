#!/usr/bin/env python3.11
"""
RTM optimization loop for SEM3D_ST7.

Workflow per accepted iteration:
  1. Initialize optimizer state if needed
  2. Compute search direction from existing gradients
  3. Propose trial materials, run forward, check Armijo; backtrack if rejected
  4. Archive accepted forward results → archive/itN/
  5. Prepare adjoint, run backward
  6. Archive backward results → archive/itN/
  7. Compute gradients for next iteration

Usage:
  python3.11 scripts/rtm_loop.py --niter 3
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

SIM_DIR = Path("SEM3D_ST7").resolve()
SCRIPTS_DIR = Path(__file__).resolve().parent


def run(cmd):
    print(f"\n$ {' '.join(str(c) for c in cmd)}")
    subprocess.run(cmd, cwd=SIM_DIR.parent, text=True, check=True)


def optimizer(*args):
    run([sys.executable, SCRIPTS_DIR / "rtm_optimizer.py", *args])


def read_state():
    return json.loads((SIM_DIR / "optim_state" / "state.json").read_text())


def clean(*names):
    for name in names:
        p = SIM_DIR / name
        if p.is_dir():
            shutil.rmtree(p)
        elif p.exists():
            p.unlink()


def swap_to_backward():
    shutil.copy2(SIM_DIR / "input.spec", SIM_DIR / "input.spec.forward.bak")
    shutil.copy2(SIM_DIR / "input_backward.spec", SIM_DIR / "input.spec")


def restore_forward_input():
    shutil.copy2(SIM_DIR / "input.spec.forward.bak", SIM_DIR / "input.spec")


def archive_forward(iter_dir: Path):
    shutil.move(SIM_DIR / "traces", iter_dir / "traces")
    shutil.move(SIM_DIR / "res", iter_dir / "res_forward")
    (iter_dir / "materials").mkdir()
    for f in ["example_la.h5", "example_mu.h5", "example_ds.h5"]:
        shutil.copy2(SIM_DIR / f, iter_dir / "materials" / f)


def archive_backward(iter_dir: Path):
    shutil.move(SIM_DIR / "res", iter_dir / "res_backward")
    shutil.copy2(SIM_DIR / "input_backward.spec", iter_dir / "input_backward.spec")
    shutil.copy2(SIM_DIR.parent / "residuals.png", iter_dir / "residuals.png")
    shutil.move(SIM_DIR / "msf", iter_dir / "msf")


def wait_for_job(job_id: str, poll_seconds: int = 10):
    print(f"Waiting for job {job_id}...")
    while True:
        p = subprocess.run(["squeue", "-h", "-j", job_id, "-o", "%T"], capture_output=True, text=True)
        state = p.stdout.strip()
        if not state:
            break
        print(f"  job {job_id}: {state}")
        time.sleep(poll_seconds)
    p = subprocess.run(["sacct", "-j", job_id, "--format=State", "--noheader"], capture_output=True, text=True)
    final = p.stdout.splitlines()[0].strip()
    print(f"Job {job_id} final state: {final}")
    if not final.startswith("COMPLETED"):
        raise RuntimeError(f"Job {job_id} did not complete: {final}")


def run_solver():
    p = subprocess.run(["sbatch", "SOLVER.sbatch"], cwd=SIM_DIR, capture_output=True, text=True, check=True)
    job_id = p.stdout.strip().split()[-1]
    print(f"Submitted job {job_id}")
    wait_for_job(job_id)


def one_iteration(max_backtracks: int, R_lambda: float, R_mu: float,
                  adaptive_regularization: bool, wp_lambda: float, wp_mu: float, lbfgs_m: int):
    reg_args = ["--R_lambda", str(R_lambda), "--R_mu", str(R_mu)]
    if adaptive_regularization:
        reg_args += ["--adaptive-regularization", "--wp_lambda", str(wp_lambda), "--wp_mu", str(wp_mu)]

    if not (SIM_DIR / "optim_state" / "state.json").exists():
        optimizer("init", *reg_args)

    it = int(read_state().get("iteration", 0)) + 1
    iter_dir = SIM_DIR / "archive" / f"it{it}"
    iter_dir.mkdir(parents=True)

    optimizer("gradient", *reg_args, "--lbfgs_m", str(lbfgs_m))

    for attempt in range(max_backtracks + 1):
        print(f"\n=== Iteration {it}, attempt {attempt} ===")
        optimizer("propose")
        clean("traces", "res", "output_forward.solver")
        restore_forward_input()
        run_solver()

        subprocess.run([sys.executable, SCRIPTS_DIR / "rtm_optimizer.py", "check", "--from_traces", *reg_args],
                       cwd=SIM_DIR.parent, text=True)
        if read_state().get("accepted"):
            print(f"Accepted iteration {it}")
            break
        print("Rejected; backtracking")
    else:
        raise RuntimeError(f"Failed to accept a step for iteration {it} after {max_backtracks + 1} attempts")

    archive_forward(iter_dir)

    restore_forward_input()
    run([sys.executable, SCRIPTS_DIR / "prepare_adjoint.py", "--sim", iter_dir / "traces"])
    swap_to_backward()

    clean("res", "output_backward.solver")
    run_solver()

    archive_backward(iter_dir)

    restore_forward_input()
    run([
        sys.executable, SCRIPTS_DIR / "compute_gradients.py",
        "--fwd_res", iter_dir / "res_forward",
        "--adj_res", iter_dir / "res_backward",
        "--outdir", SIM_DIR / "gradients",
        "--plot_output", SIM_DIR / "gradients.png",
    ])

    print(f"\n=== Completed iteration {it}, archived in: {iter_dir} ===")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--niter", type=int, default=1)
    parser.add_argument("--max-backtracks", type=int, default=50)    
    parser.add_argument("--R_lambda", type=float, default=1e-4)
    parser.add_argument("--R_mu", type=float, default=1e-4)
    parser.add_argument("--adaptive-regularization", action="store_true")
    parser.add_argument("--wp_lambda", type=float, default=0.5)
    parser.add_argument("--wp_mu", type=float, default=0.5)    
    parser.add_argument("--lbfgs_m", type=int, default=5)
    args = parser.parse_args()
    
    for _ in range(args.niter):
        one_iteration(args.max_backtracks, args.R_lambda, args.R_mu, args.adaptive_regularization, args.wp_lambda, args.wp_mu, args.lbfgs_m)


if __name__ == "__main__":
    main()
