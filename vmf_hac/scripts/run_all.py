import subprocess

EXPERIMENT_MODULES = [
    "vmf_hac.experiments.explore_datasets",
    "vmf_hac.experiments.explore_dimensions",
    "vmf_hac.experiments.explore_embedding_models",
    "vmf_hac.experiments.explore_gamma",
    "vmf_hac.experiments.explore_vmf_ward",
    "vmf_hac.experiments.explore_gamma_linkage_behavior",
]

PLOTTING_MODULES = [
    "vmf_hac.plotting.datasets",
    "vmf_hac.plotting.gamma",
    "vmf_hac.plotting.dimensions",
    "vmf_hac.plotting.correlation_vmf_ward",
    "vmf_hac.plotting.gamma_linkage_behavior",
    "vmf_hac.plotting.surface",
]


def _run_modules(modules: list[str], kind: str) -> None:
    failed_modules: list[str] = []
    for module in modules:
        print(f"Running {module}...")
        result = subprocess.run(["uv", "run", "python", "-m", module], check=False)
        if result.returncode != 0:
            failed_modules.append(module)
    if failed_modules:
        failed = ", ".join(failed_modules)
        msg = f"One or more {kind} modules failed: {failed}"
        raise RuntimeError(msg)


def run_all_experiments() -> None:
    _run_modules(EXPERIMENT_MODULES, "experiment")


def run_all_plots() -> None:
    _run_modules(PLOTTING_MODULES, "plotting")
