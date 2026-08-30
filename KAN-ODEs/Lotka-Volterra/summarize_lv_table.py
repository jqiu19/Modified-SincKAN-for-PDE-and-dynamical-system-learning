#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def read_best_test(path: Path) -> float:
    with h5py.File(path, "r") as f:
        loss = np.asarray(f["loss"]).squeeze()
        test = np.asarray(f["loss_test"]).squeeze()
    mask = loss != 0
    if not np.any(mask):
        raise ValueError(f"no nonzero loss entries in {path}")
    return float(np.min(test[mask]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", help="model=glob, e.g. kanode='results_lv_table_kanode_*'")
    args = parser.parse_args()

    rows = []
    base = Path(__file__).resolve().parent
    for spec in args.runs:
        model, pattern = spec.split("=", 1)
        vals = []
        for run_dir in sorted(base.glob(pattern)):
            mats = sorted((run_dir / "checkpoints").glob("LV_*_results.mat"))
            if not mats:
                continue
            vals.append(read_best_test(mats[0]))
        arr = np.asarray(vals, dtype=float)
        if len(arr) == 0:
            rows.append((model, "missing", "missing", 0))
        else:
            rows.append((model, f"{arr.mean():.6e}", f"{arr.std(ddof=1):.6e}" if len(arr) > 1 else "nan", len(arr)))

    print("| Equation | " + " | ".join(row[0] for row in rows) + " |")
    print("|---|" + "|".join(["---:"] * len(rows)) + "|")
    vals = []
    for _, mean, std, n in rows:
        vals.append("missing" if n == 0 else f"{mean} ± {std}")
    print("| Lotka-Volterra ODE system | " + " | ".join(vals) + " |")
    print()
    for model, mean, std, n in rows:
        print(f"{model}: n={n}, mean={mean}, std={std}")


if __name__ == "__main__":
    main()
