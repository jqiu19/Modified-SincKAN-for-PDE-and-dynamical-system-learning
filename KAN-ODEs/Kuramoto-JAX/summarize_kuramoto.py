import argparse
import json
from pathlib import Path

import numpy as np


def fmt(mean, std):
    return f"{mean:.3e} ± {std:.2e}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--metric", default="test_mse")
    parser.add_argument("--models", nargs="+", default=["mlp", "modifiedmlp", "kan", "sinckan", "modifiedsinckan"])
    args = parser.parse_args()

    root = Path(args.results_dir)
    row = {"Equation": "Kuramoto oscillator network"}
    details = {}
    for model in args.models:
        vals = []
        runs = sorted(root.glob(f"{model}*/summary.json"))
        for path in runs:
            data = json.loads(path.read_text())
            if args.metric in data.get("best", {}):
                vals.append(float(data["best"][args.metric]))
        if vals:
            arr = np.asarray(vals, dtype=float)
            row[model] = fmt(arr.mean(), arr.std(ddof=1) if arr.size > 1 else 0.0)
            details[model] = vals
        else:
            row[model] = "missing"
            details[model] = []

    headers = ["Equation", "MLP", "Modified MLP", "KAN", "SincKAN", "Ours"]
    keys = ["Equation", "mlp", "modifiedmlp", "kan", "sinckan", "modifiedsinckan"]
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    print("| " + " | ".join(row[k] for k in keys) + " |")
    print("\nRaw values:")
    print(json.dumps(details, indent=2))


if __name__ == "__main__":
    main()
