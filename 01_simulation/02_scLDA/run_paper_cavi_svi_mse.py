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
    cavi_max_iter=10,
    svi_max_steps=100,
    batch_size=1000,
    tau0=1.0,
    kappa=0.6,
    eval_every_cavi=1,
    eval_every_svi=10,
    eval_size=1_000,
)


def run_one_cavi_svi_seed(seed: int) -> pd.DataFrame:
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
    cavi = model.CAVI(
        train_counts,
        test_counts,
        sim.beta_true,
        lambda_init=lambda_init,
        max_iter=PAPER_KWARGS["cavi_max_iter"],
        evaluate_every=PAPER_KWARGS["eval_every_cavi"],
        eval_indices=eval_indices,
    )["history"]
    svi = model.SVI(
        train_counts,
        test_counts,
        sim.beta_true,
        lambda_init=lambda_init,
        max_steps=PAPER_KWARGS["svi_max_steps"],
        batch_size=PAPER_KWARGS["batch_size"],
        tau0=PAPER_KWARGS["tau0"],
        kappa=PAPER_KWARGS["kappa"],
        evaluate_every=PAPER_KWARGS["eval_every_svi"],
        eval_indices=eval_indices,
        seed=seed + 40_000,
    )["history"]
    history = pd.concat([cavi, svi], ignore_index=True)
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
    parser.add_argument("--n-jobs", type=int, default=20)
    parser.add_argument(
        "--pyro-history",
        type=Path,
        default=RESULTS_DIR / "history_paper_pyro.csv",
    )
    parser.add_argument(
        "--out-history",
        type=Path,
        default=RESULTS_DIR / "history.csv",
    )
    args = parser.parse_args()

    seeds = list(range(args.seed_start, args.seed_start + args.n_reps))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[pd.DataFrame] = []

    if args.n_jobs == 1:
        for seed in seeds:
            print(f"cavi/svi seed={seed}", flush=True)
            rows.append(run_one_cavi_svi_seed(seed))
    else:
        with ProcessPoolExecutor(max_workers=args.n_jobs) as pool:
            futures = {pool.submit(run_one_cavi_svi_seed, seed): seed for seed in seeds}
            for fut in as_completed(futures):
                seed = futures[fut]
                hist = fut.result()
                rows.append(hist)
                print(f"done cavi/svi seed={seed} rows={len(hist)}", flush=True)

    cavi_svi = pd.concat(rows, ignore_index=True)
    out_cs = RESULTS_DIR / "history_paper_cavi_svi_mse.csv"
    cavi_svi.to_csv(out_cs, index=False)

    pieces = [cavi_svi]
    if args.pyro_history.is_file():
        pyro = pd.read_csv(args.pyro_history)
        if "runtime" not in pyro.columns and "time_sec" in pyro.columns:
            pyro["runtime"] = pyro["time_sec"]
        if "time_sec" not in pyro.columns and "runtime" in pyro.columns:
            pyro["time_sec"] = pyro["runtime"]
        pyro["method"] = pyro["method"].astype(str).str.upper()
        # Align pyro seeds to the CAVI/SVI seed set when possible.
        pyro = pyro.loc[pyro["seed"].isin(seeds)].copy()
        pieces.append(pyro)

    merged = pd.concat(pieces, ignore_index=True, sort=False)
    merged.to_csv(args.out_history, index=False)
    print("wrote", out_cs)
    print("wrote", args.out_history)
    print(merged.groupby("method")[["topic_mse", "topic_tv"]].agg(["count", "mean"]))


if __name__ == "__main__":
    main()
