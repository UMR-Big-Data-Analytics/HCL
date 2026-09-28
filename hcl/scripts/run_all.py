import subprocess

EXPERIMENT_MODULES = [
    "hlc.experiments.explore_datasets",
    "hlc.experiments.explore_dimensions",
    "hlc.experiments.explore_embedding_models",
    "hlc.experiments.explore_gamma",
    "hlc.experiments.explore_vmf_ward",
    "hlc.experiments.explore_gamma_linkage_behavior",
]

PLOTTING_MODULES = [
    "hlc.plotting.dataset",
    "hlc.plotting.gamma",
    "hlc.plotting.dimensions",
    "hlc.plotting.correlation_hcl_ward",
    "hlc.plotting.gamma_linkage_behavior",
    "hlc.plotting.surface",
    "hlc.plotting.embedding_models",
    "hlc.plotting.global_population",
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
