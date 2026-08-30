import argparse
import subprocess
import sys
import time
from pathlib import Path


BENCHMARKS = {
    "bl": Path("bl.sh"),
    "fractal": Path("fractal.sh"),
    "steady_poisson": Path("../pde/steady_poisson.sh"),
}


def main():
    parser = argparse.ArgumentParser(description="Run modified-SincKAN benchmark shells on the local RTX 3070.")
    parser.add_argument("--suite", type=str, default="all", help="benchmark suite name or 'all'")
    parser.add_argument("--time_budget_hours", type=float, default=24.0, help="stop after this many hours")
    parser.add_argument("--python_cmd", type=str, default="python", help="python launcher prefix, e.g. 'conda run -n jingwei python'")
    args = parser.parse_args()

    start = time.time()
    suites = BENCHMARKS.keys() if args.suite == "all" else [args.suite]
    for suite in suites:
        if suite not in BENCHMARKS:
            raise ValueError(f"Unknown suite: {suite}")
        elapsed_hours = (time.time() - start) / 3600.0
        if elapsed_hours >= args.time_budget_hours:
            print(f"Time budget reached after {elapsed_hours:.2f} hours.")
            break
        script_path = BENCHMARKS[suite]
        print(f"Running suite {suite} via {script_path}")
        for line in script_path.read_text(encoding="utf-8").splitlines():
            command = line.strip()
            if not command or command.startswith("#"):
                continue
            if command.startswith("python "):
                command = f"{args.python_cmd} {command[len('python '):]}"
            print(f"  -> {command}")
            subprocess.run(command, check=True, shell=True)

    subprocess.run(
        f"{args.python_cmd} summarize_table1.py --search_dir .",
        check=True,
        shell=True,
    )


if __name__ == "__main__":
    main()
