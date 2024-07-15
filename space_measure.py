import numpy as np

from multiprocessing import Pool
from pathlib import Path

import fpfz


workdir = Path('space_measure')
workdir.mkdir(parents=True, exist_ok=True)


if __name__ == '__main__':
    ns = 2 ** np.arange(3, 28)

    params_rec = [(2,),(3,),(7,),]
    params_rek = [(7, 4), (7, 14)]
    params_egh = [(7,)]

    def extend_params(ps, n):
        return [(n, *p, workdir) for p in ps]

    futures = []
    with Pool() as pool:
        for n in ns:  # this way jobs are submitted from low n to high
            for method in (
                fpfz.calc_space_egh,
            ):
                futures.append(pool.starmap_async(method, extend_params(params_egh, n)))

            for method in (
                fpfz.calc_space_rec_k,
            ):
                futures.append(pool.starmap_async(method, extend_params(params_rek, n)))

            for method in (
                fpfz.calc_space_rec,
            ):
                futures.append(pool.starmap_async(method, extend_params(params_rec, n)))

        for f in futures:
            f.wait()
