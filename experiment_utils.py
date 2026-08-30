import csv
import time
from pathlib import Path


def build_results_path(args, prefix="results"):
    result_tag = getattr(args, "result_tag", "")
    if result_tag:
        return f"{prefix}_{args.network}_{args.datatype}_{args.seed}_{result_tag}.csv"
    return f"{prefix}_{args.network}_{args.datatype}_{args.seed}_.csv"


def _is_smoketest_results_file(save_path: Path):
    if not save_path.is_file():
        return False
    try:
        with save_path.open("r", newline="") as f:
            rows = list(csv.DictReader(f))
    except OSError:
        return False
    if not rows:
        return False
    total_ite = rows[-1].get("total_ite")
    if not total_ite:
        return False
    try:
        return int(float(total_ite)) < 100
    except ValueError:
        return False


def _fallback_results_path(save_path: Path):
    return save_path.with_name(f"{save_path.stem}_rerun{save_path.suffix}")


def append_result_row(save_here, header, row):
    save_path = Path(save_here)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if _is_smoketest_results_file(save_path):
        try:
            save_path.unlink()
        except PermissionError:
            save_path = _fallback_results_path(save_path)
    last_error = None
    for _ in range(5):
        try:
            file_exists = save_path.is_file()
            with save_path.open("a", newline="") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(header)
                writer.writerow(row)
            return
        except PermissionError as error:
            last_error = error
            time.sleep(1.0)
    fallback_path = _fallback_results_path(save_path)
    file_exists = fallback_path.is_file()
    with fallback_path.open("a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(row)


def init_wandb_run(args, script_name, extra_config=None):
    if getattr(args, "wandb", 1) == 0:
        return None

    try:
        import wandb
    except ImportError:
        print("wandb is not installed; skipping remote logging.")
        return None

    config = vars(args).copy()
    if extra_config:
        config.update(extra_config)

    run = wandb.init(
        project=getattr(args, "wandb_project", "modifiedsinckan"),
        group=getattr(args, "wandb_group", script_name),
        job_type=script_name,
        name=f"{args.network}-{args.datatype}-seed{args.seed}",
        config=config,
        reinit=True,
    )
    return run


def log_wandb(run, metrics, step=None):
    if run is None:
        return
    if step is None:
        run.log(metrics)
    else:
        run.log(metrics, step=step)


def finish_wandb(run, summary=None):
    if run is None:
        return
    if summary:
        for key, value in summary.items():
            run.summary[key] = value
    run.finish()
