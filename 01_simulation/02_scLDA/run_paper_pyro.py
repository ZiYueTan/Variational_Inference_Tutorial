#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import sys

for _thread_var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_thread_var, "1")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results" / "scLDA"
TUTORIAL_CODE = Path("/home/byual/VI_tutorial/Tutorial_code/VI_simu/scLDA")
HERE = Path(__file__).resolve().parent

sys.path.insert(0, str(HERE))
sys.path.insert(0, str(TUTORIAL_CODE))

from LDA import LDA  # noqa: E402
from sc_lda_methods import (  # noqa: E402
    initialize_lambda,
    simulate_sc_lda_data,
    train_test_split_counts,
)


PAPER_KWARGS = dict(
    n_cells=10_000,
    n_genes=5_000,
    n_topics=10,
    mean_library_size=500.0,
    eta_true=0.05,
    alpha_true=0.2,
    eta_fit=0.05,
    alpha_fit=0.2,
    train_fraction=0.8,
    # Longer budget than CAVI/SVI (100k cells): Pyro is GPU-fast, so train to 1M
    # processed cells; the redraw script still caps cell-axis panels at 100k.
    pyro_max_steps=1000,
    batch_size=1000,
    pyro_lr=0.1,
    pyro_lrd=0.998,
    pyro_num_particles=1,
    eval_every_pyro=10,
    eval_size=1_000,
)


def run_one_pyro_seed(
    seed: int,
    device: str = "cuda",
    pyro_max_steps: int | None = None,
    eval_every_pyro: int | None = None,
) -> pd.DataFrame:
    max_steps = PAPER_KWARGS["pyro_max_steps"] if pyro_max_steps is None else pyro_max_steps
    eval_every = PAPER_KWARGS["eval_every_pyro"] if eval_every_pyro is None else eval_every_pyro
    sim = simulate_sc_lda_data(
        n_cells=PAPER_KWARGS["n_cells"],
        n_genes=PAPER_KWARGS["n_genes"],
        n_topics=PAPER_KWARGS["n_topics"],
        mean_library_size=PAPER_KWARGS["mean_library_size"],
        eta_true=PAPER_KWARGS["eta_true"],
        alpha_true=PAPER_KWARGS["alpha_true"],
        seed=seed,
    )
    train_counts, test_counts = train_test_split_counts(
        sim.counts, train_fraction=PAPER_KWARGS["train_fraction"], seed=seed + 10_000
    )
    lambda_init = initialize_lambda(
        train_counts,
        n_topics=PAPER_KWARGS["n_topics"],
        eta=PAPER_KWARGS["eta_fit"],
        seed=seed + 20_000,
    )
    rng = np.random.default_rng(seed + 30_000)
    eval_indices = np.sort(
        rng.choice(PAPER_KWARGS["n_cells"], size=PAPER_KWARGS["eval_size"], replace=False)
    )

    model = LDA(
        n_topics=PAPER_KWARGS["n_topics"],
        alpha=PAPER_KWARGS["alpha_fit"],
        eta=PAPER_KWARGS["eta_fit"],
        seed=seed + 20_000,
    )
    out = model.Pyro(
        train_counts,
        test_counts,
        sim.beta_true,
        lambda_init=lambda_init,
        max_steps=max_steps,
        batch_size=PAPER_KWARGS["batch_size"],
        pyro_lr=PAPER_KWARGS["pyro_lr"],
        pyro_lrd=PAPER_KWARGS["pyro_lrd"],
        pyro_num_particles=PAPER_KWARGS["pyro_num_particles"],
        evaluate_every=eval_every,
        eval_indices=eval_indices,
        seed=seed + 40_000,
        device=device,
    )
    history = out["history"].copy()
    history.insert(0, "seed", int(seed))
    history["n_cells"] = PAPER_KWARGS["n_cells"]
    history["n_genes"] = PAPER_KWARGS["n_genes"]
    history["n_topics"] = PAPER_KWARGS["n_topics"]
    history["eval_size"] = PAPER_KWARGS["eval_size"]
    history["epoch"] = history["processed_cells"] / PAPER_KWARGS["n_cells"]
    history["time_sec"] = history["runtime"]
    return history


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-reps", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=2,
        help="Parallel GPU workers. Keep small (1-2) to avoid CUDA OOM.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Pyro device (cuda recommended for paper-scale runs). "
        "With --n-jobs>1 and device=cuda, workers round-robin across visible GPUs.",
    )
    parser.add_argument(
        "--base-history",
        type=Path,
        default=RESULTS_DIR / "history_paper_cavi_svi_mse.csv",
        help="CAVI/SVI history to merge with (prefer the *_mse.csv re-run).",
    )
    parser.add_argument(
        "--out-history",
        type=Path,
        default=RESULTS_DIR / "history.csv",
    )
    parser.add_argument("--pyro-max-steps", type=int, default=PAPER_KWARGS["pyro_max_steps"])
    parser.add_argument("--eval-every", type=int, default=PAPER_KWARGS["eval_every_pyro"])
    args = parser.parse_args()

    seeds = list(range(args.seed_start, args.seed_start + args.n_reps))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pyro_rows: list[pd.DataFrame] = []

    devices: list[str]
    if args.device == "cuda" and args.n_jobs > 1:
        try:
            import torch

            n_gpu = int(torch.cuda.device_count())
        except Exception:
            n_gpu = 0
        devices = [f"cuda:{i}" for i in range(max(n_gpu, 1))] if n_gpu else [args.device]
    else:
        devices = [args.device]

    print(
        f"pyro_max_steps={args.pyro_max_steps} eval_every={args.eval_every} "
        f"batch={PAPER_KWARGS['batch_size']} -> max_cells="
        f"{args.pyro_max_steps * PAPER_KWARGS['batch_size']} "
        f"n_reps={args.n_reps} devices={devices}",
        flush=True,
    )

    if args.n_jobs == 1:
        for seed in seeds:
            device = devices[(seed - args.seed_start) % len(devices)]
            print(f"pyro seed={seed} device={device}", flush=True)
            pyro_rows.append(
                run_one_pyro_seed(
                    seed,
                    device=device,
                    pyro_max_steps=args.pyro_max_steps,
                    eval_every_pyro=args.eval_every,
                )
            )
    else:
        with ProcessPoolExecutor(max_workers=args.n_jobs) as pool:
            futures = {
                pool.submit(
                    run_one_pyro_seed,
                    seed,
                    devices[(seed - args.seed_start) % len(devices)],
                    args.pyro_max_steps,
                    args.eval_every,
                ): seed
                for seed in seeds
            }
            for fut in as_completed(futures):
                seed = futures[fut]
                hist = fut.result()
                pyro_rows.append(hist)
                print(f"done pyro seed={seed} rows={len(hist)}", flush=True)

    pyro = pd.concat(pyro_rows, ignore_index=True)
    pyro_path = RESULTS_DIR / "history_paper_pyro.csv"
    pyro.to_csv(pyro_path, index=False)

    base = pd.read_csv(args.base_history)
    if "runtime" not in base.columns and "time_sec" in base.columns:
        base["runtime"] = base["time_sec"]
    if "time_sec" not in base.columns and "runtime" in base.columns:
        base["time_sec"] = base["runtime"]
    base["method"] = base["method"].astype(str).str.upper()
    # Drop any previous PYRO rows before merging.
    base = base.loc[base["method"] != "PYRO"].copy()

    merged = pd.concat([base, pyro], ignore_index=True, sort=False)
    merged.to_csv(args.out_history, index=False)
    print("wrote", pyro_path)
    print("wrote", args.out_history)
    print(merged.groupby("method").size())


if __name__ == "__main__":
    main()
