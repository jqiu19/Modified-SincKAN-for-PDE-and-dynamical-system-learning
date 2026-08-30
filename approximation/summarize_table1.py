import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


def is_smoketest_row(row):
    source = Path(row.get("_source", "")).name.lower()
    if "smoketest" in source:
        return True
    total_ite = row.get("total_ite")
    if not total_ite:
        return False
    try:
        return int(float(total_ite)) < 100
    except ValueError:
        return False


def load_rows(search_dir: Path):
    rows = []
    for csv_path in sorted(search_dir.rglob("results_*.csv")):
        mtime = csv_path.stat().st_mtime
        with csv_path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row["_source"] = str(csv_path)
                row["_mtime"] = mtime
                if not is_smoketest_row(row):
                    rows.append(row)
    return rows


def dedupe_rows(rows):
    best_rows = {}
    for row in rows:
        key = (row.get("datatype"), row.get("network"), row.get("seed"))
        total_ite = float(row.get("total_ite", 0) or 0)
        current = best_rows.get(key)
        if current is None:
            best_rows[key] = row
            continue
        current_total_ite = float(current.get("total_ite", 0) or 0)
        if total_ite > current_total_ite:
            best_rows[key] = row
        elif total_ite == current_total_ite and row["_mtime"] > current["_mtime"]:
            best_rows[key] = row
    return list(best_rows.values())


def pick_rmse(row):
    if row.get("test_rmse"):
        return float(row["test_rmse"])
    if row.get("rmse"):
        return float(row["rmse"])
    if row.get("test_mse"):
        return math.sqrt(float(row["test_mse"]))
    if row.get("mse"):
        return math.sqrt(float(row["mse"]))
    raise KeyError("No RMSE/MSE column found in row.")


def main():
    parser = argparse.ArgumentParser(description="Summarize RMSE results for modified-SincKAN Table 1.")
    parser.add_argument("--search_dir", type=str, default=".", help="root directory containing results_*.csv")
    parser.add_argument("--output_csv", type=str, default="table1_modifiedsinckan.csv", help="output csv path")
    parser.add_argument("--output_md", type=str, default="table1_modifiedsinckan.md", help="output markdown path")
    args = parser.parse_args()

    rows = dedupe_rows(load_rows(Path(args.search_dir)))
    grouped = defaultdict(list)
    for row in rows:
        try:
            grouped[(row["datatype"], row["network"])].append(pick_rmse(row))
        except KeyError:
            continue

    summary_rows = []
    for (datatype, network), values in sorted(grouped.items()):
        summary_rows.append({
            "datatype": datatype,
            "network": network,
            "num_seeds": len(values),
            "rmse_mean": float(np.mean(values)),
            "rmse_std": float(np.std(values)),
        })

    with Path(args.output_csv).open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["datatype", "network", "num_seeds", "rmse_mean", "rmse_std"],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    md_lines = [
        "| datatype | network | seeds | rmse_mean | rmse_std |",
        "|---|---|---:|---:|---:|",
    ]
    for row in summary_rows:
        md_lines.append(
            f"| {row['datatype']} | {row['network']} | {row['num_seeds']} | "
            f"{row['rmse_mean']:.4e} | {row['rmse_std']:.4e} |"
        )
    Path(args.output_md).write_text("\n".join(md_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
