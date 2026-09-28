import argparse
import json
import shlex
import subprocess
from collections.abc import Generator

from hcl.definitions import ROOT_DIR


def iterator() -> Generator[tuple[str, str]]:
    config_path = ROOT_DIR / "config.json"
    with config_path.open() as f:
        config = json.load(f)
    for model in config["models"]:
        for dataset_id in config["datasets"]:
            yield model, dataset_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submit one embedding job per model/dataset combination.",
    )
    parser.add_argument(
        "--mode",
        choices=("sbatch", "direct"),
        default="sbatch",
        help="sbatch submits one scheduler job per combination; direct starts all processes locally in parallel.",
    )
    parser.add_argument(
        "--sbatch-args",
        nargs="*",
        default=[],
        help=(
            "Extra arguments passed to sbatch (only used when --mode sbatch). "
            "Supports either separate args (--sbatch-args --time=... --partition=...) "
            "or a single quoted string."
        ),
    )
    return parser.parse_args()


def build_job_command(model: str, dataset: str) -> list[str]:
    return [
        "uv",
        "run",
        "python",
        "-m",
        "hlc.scripts.generate_embeddings",
        "--model",
        model,
        "--dataset",
        dataset,
    ]


def normalize_sbatch_args(raw_args: list[str]) -> list[str]:
    normalized: list[str] = []
    for arg in raw_args:
        normalized.extend(shlex.split(arg))
    return normalized


def run_direct() -> None:
    processes: list[tuple[str, str, subprocess.Popen[str]]] = []
    for model, dataset in iterator():
        cmd = build_job_command(model, dataset)
        process = subprocess.Popen(cmd, text=True)
        processes.append((model, dataset, process))

    failed_jobs: list[tuple[str, str, int]] = []
    for model, dataset, process in processes:
        return_code = process.wait()
        if return_code != 0:
            failed_jobs.append((model, dataset, return_code))

    if failed_jobs:
        formatted = ", ".join(f"{model}/{dataset} (exit {code})" for model, dataset, code in failed_jobs)
        msg = f"One or more direct jobs failed: {formatted}"
        raise RuntimeError(msg)


def run_sbatch(args: argparse.Namespace) -> None:
    sbatch_args = normalize_sbatch_args(args.sbatch_args)
    for model, dataset in iterator():
        job_cmd = build_job_command(model, dataset)
        submit_cmd = ["sbatch", *sbatch_args, "--wrap", shlex.join(job_cmd)]
        subprocess.run(submit_cmd, check=True)


def main() -> None:
    args = parse_args()
    if args.mode == "sbatch":
        run_sbatch(args)
    else:
        run_direct()


if __name__ == "__main__":
    main()
