from itertools import repeat, starmap
from collections import Counter
from multiprocessing import Pool as MpPool
from random import sample

import numpy as np
from numpy.random import default_rng

import met

rng = default_rng()


def extra_cells(n_deg, n_mem):
    ibf_cell_bits = log_u * 2
    deg_bits = (n_deg + n_mem + 3) * np.log2(log_u)
    return int(np.ceil(deg_bits / ibf_cell_bits))


def partition_m(m, parts):
    m_each, remainder = np.divmod(m, parts)
    m_cells = np.repeat(m_each, parts)
    m_cells[:remainder] += 1
    assert sum(m_cells) == m
    return m_cells


rep = 1000
log_u = 32
dist = Counter({
    .05: rep,
    # .25: rep,
    # .50: rep,
    # .75: rep,
    .95: rep,
})
p = np.fromiter(dist.keys(), dtype=float)
c = np.fromiter(dist.values(), dtype=int)
load = 1.23

expected_set_size = p @ c
m = int(expected_set_size * load)

deg = np.array([
    [1, 1],
    [1, 1],
    [1, 1],
])
m_cells = partition_m(m - extra_cells(deg.size, len(deg)), len(deg))
n_cell_types, n_data_types = deg.shape


def pack(deg, m_cells):
    return np.append(deg.ravel(), m_cells.ravel())


def unpack(x):
    deg, m_cells = np.split(x, [n_cell_types * n_data_types, ])
    deg = deg.reshape((n_cell_types, n_data_types))
    return deg.astype(int), m_cells.astype(int)


def simulate_gap_once(deg, m_cells):
    n = rng.binomial(c, p)

    s = sample(range(2 ** log_u), n.sum())
    keys_by_p = np.array_split(s, n.cumsum()[:-1])

    key2p = {
        key: prob
        for prob, keys in zip(p, keys_by_p)
        for key in keys
    }

    def key2type(key):
        return (p >= key2p[key]).argmax()

    t = met.METIBF(deg_matrix=deg, m_cells=m_cells, key2type=key2type)
    t.insert_from(s)
    return 1 - len(t.peel()) / len(key2p)


class LocalPool:
    starmap = starmap

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def cost(x, n_sim=500, multiprocess=True):
    x = x.astype(int)

    deg, m_cells = unpack(x)

    Pool = MpPool if multiprocess else LocalPool

    with Pool() as p:
        avg_gap = sum(p.starmap(simulate_gap_once, repeat((deg, m_cells), n_sim))) / n_sim

    return avg_gap


class MyTakeStep:
    def __init__(self, n_cell_types, n_data_types, stepsize=0.5):
        self.stepsize = stepsize
        self.rng = np.random.default_rng()
        self.n_cell_types = n_cell_types
        self.n_data_types = n_data_types

    def mem_step(self, x):
        i_from, i_to = rng.choice(self.n_cell_types, 2) + self.n_data_types * self.n_cell_types
        amount = int(np.floor(x[i_from] * self.stepsize / 2))
        x[i_from] -= amount
        x[i_to] += amount

    def deg_step(self, x):
        i_top = self.n_data_types * self.n_cell_types
        x[:i_top] += rng.normal(.1, .5 + 6 * self.stepsize, i_top).astype(int)

    def __call__(self, x):
        self.mem_step(x)
        self.deg_step(x)
        x = x.clip(0).astype(int)
        return x

