import numpy as np

from multiprocessing import Pool
from pathlib import Path

from psutil import cpu_count

import fpfz


workdir = Path('measuring_clean')
workdir.mkdir(parents=True, exist_ok=True)


def ready_future_idxs(futures: list) -> list:
    return [i for i, f in enumerate(futures) if f.ready()]


def clear_at_least_one(futures: list) -> int:
    ready_i_reversed = sorted(ready_future_idxs(futures), reverse=True)

    if futures and not ready_i_reversed:
        # if no one is ready, wait for the oldest one
        ready_i_reversed.append(0)

    for i in ready_i_reversed:
        try:
            futures.pop(i).get()
        except Exception as e:
            print(f'Exeption: {e}')

    return len(ready_i_reversed)


if __name__ == '__main__':
    ns = 2 ** np.arange(3, 28)
    reps = 30
    sample_size = 10**6

    params_rec = [(7,)]
    params_rek = [(7, 14)]
    params_egh = [(7,)]

    n_workers = max(1, 7*cpu_count(logical=False) // 8)  # avoid overload, use only physical CPUs

    def extend_params(ps, n):
        return [(n, *p, workdir, reps, sample_size) for p in ps]

    jobs = []
    for n in reversed(ns):  # this way jobs are submitted from low n to high
        for method in (
            # fpfz.measure_egh_matrix,
            # fpfz.measure_egh_sparse_matrix,
            # fpfz.measure_egh_column_generator,
            fpfz.measure_egh_column_generator_clean,
        ):
            jobs.extend((method, p) for p in extend_params(params_egh, n))

        for method in (
            # fpfz.measure_k_matrix,
            # fpfz.measure_k_sparse_matrix,
            # fpfz.measure_k_column_generator,
            fpfz.measure_k_column_generator_clean,
        ):
            jobs.extend((method, p) for p in extend_params(params_rek, n))

        for method in (
            # fpfz.measure_rec_matrix,
            # fpfz.measure_rec_sparse_matrix,
            # fpfz.measure_rec_column_generator,
            fpfz.measure_rec_column_generator_clean,
        ):
            jobs.extend((method, p) for p in extend_params(params_rec, n))

    print(f'#jobs={len(jobs)}')

    futures = []
    with Pool(processes=n_workers, maxtasksperchild=1) as pool:
        while jobs:
            print('=' * 20, f'{len(jobs)=}', '=' * 20)
            if len(futures) >= n_workers:
                clear_at_least_one(futures)

            while jobs and len(futures) < n_workers:
                futures.append(pool.apply_async(*jobs.pop()))

        while futures:
            clear_at_least_one(futures)
