from collections.abc import Callable, Sequence
from typing import Literal

from joblib import Parallel
from tqdm import tqdm

# noinspection PyTypeHints
type Job[**P, V] = tuple[Callable[P, V], P.args, P.kwargs]


def task[**P, V](func: Callable[P, V]) -> Callable[P, Job[P, V]]:
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> Job[P, V]:
        return func, args, kwargs

    return wrapper


def gather[**P, V](
    jobs: Sequence[Job[P, V]],
    n_jobs: int = -1,
    batch_size: int = 1,
    show_progress: bool = False,
    backend: Literal["loky", "multiprocessing", "sequential", "threading"] | None = None,
) -> list[V]:
    if show_progress:
        par = Parallel(n_jobs=n_jobs, backend=backend, batch_size=batch_size, return_as="generator")(jobs)
        return list(tqdm(par, total=len(jobs)))

    return Parallel(n_jobs=n_jobs, backend=backend, batch_size=batch_size)(jobs)
