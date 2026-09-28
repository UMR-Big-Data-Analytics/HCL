import subprocess

EXPERIMENT_MODULES = [
    "hcl.experiments.explore_datasets",
    "hcl.experiments.explore_dimensions",
    "hcl.experiments.explore_embedding_models",
    "hcl.experiments.explore_gamma",
    "hcl.experiments.explore_hcl_ward",
    "hcl.experiments.explore_gamma_linkage_behavior",
]

PLOTTING_MODULES = [
    "hcl.plotting.dataset",
    "hcl.plotting.gamma",
    "hcl.plotting.dimensions",
    "hcl.plotting.correlation_hcl_ward",
    "hcl.plotting.gamma_linkage_behavior",
    "hcl.plotting.surface",
    "hcl.plotting.embedding_models",
    "hcl.plotting.global_population",
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
