import argparse
import itertools
import math
import operator
import pickle
import random
import shutil
import sys
from abc import abstractmethod, ABC
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from multiprocessing import Pool, cpu_count
from os import getpid
from pathlib import Path
from time import process_time_ns, sleep
from typing import Collection, Any

import attr
import numpy as np
import pandas as pd
from mmh3 import hash as mmh3

import IBLT_success_rates
import primes
from cache import cache
from itertools_recipes import first_true, tail, take


if not hasattr(math, 'comb'):
    def math_comb(n, r):
        f = math.factorial
        return f(n) // (f(r) * f(n - r))

    math.comb = math_comb


def min_binom(target,k):
    # log-scale range filter
    def min_binom_log_range(target,k):
        n=1
        while binom(n,k)<target:
            n*=2
        return range(n//2,n+1)

    for n in min_binom_log_range(target,k):
        if binom(n,k)>=target:
            break
    return n


@attr.s
class MatrixDecoder:
    matrix: np.ndarray = attr.ib()

    def is_decodable(self, elements: Collection[int]) -> bool:
        to_peel = set(elements)

        if len(to_peel) < len(elements):
            # easy case, stopping set of size 2
            return False

        while len(to_peel) != 0:
            to_remove = self.peel(to_peel)
            if len(to_remove) == 0:
                return len(to_peel) == 0
            to_peel.difference_update(to_remove)

        return len(to_peel) == 0

    def peel(self, to_peel):
        idx_to_element = np.fromiter(to_peel, dtype=int)
        codewords = self.matrix[idx_to_element]
        counters = codewords.sum(0)
        pures = [i for i, c in enumerate(counters) if c == 1]
        return idx_to_element[list(set(codewords[:, pures].nonzero()[0]))]


class MatrixAux:
    type = np.int8

    @staticmethod
    def I(n):
        return np.eye(n, dtype=MatrixAux.type)

    @staticmethod
    def F0(n, m):
        return np.zeros((n, m), dtype=MatrixAux.type)

    @staticmethod
    def F1(n, m):
        return np.ones((n, m), dtype=MatrixAux.type)

    @staticmethod
    def C0(n):
        return MatrixAux.F0(n, 1)

    @staticmethod
    def C1(n):
        return MatrixAux.F1(n, 1)


class Steiner:
    @staticmethod
    def m(q, a):
        return q**a + 1

    @staticmethod
    def n(q, m):
        return round((m**3 - 3*m**2 + 2*m) / (q*(q**2 - 1)))

    @staticmethod
    def generate_df(pg, max_m):
        l = []
        for q in itertools.islice(primes.Power(pg), 5, max_m):
            for a in itertools.count(2):
                m = Steiner.m(q, a)
                if m > max_m:
                    break
                l.append((q, a, Steiner.n(q, m), m))

        df = pd.DataFrame(l, columns=('q', 'a', 'n', 'm'))
        df['d'] = ((df.q + 1) / 2).apply(np.ceil).astype(int)
        df['k'] = df.q + 1
        return df


@attr.s
class UniverseSizeCalculator:
    pg = attr.ib(factory=primes.Generator)

    @staticmethod
    def d2_construction(m_bits, k=None):
        if k is None:
            return 2**m_bits - 1
        return math.comb(m_bits, k)

    @staticmethod
    def d3_construction(m_bits):
        """d=3 here is for PD"""
        m = m_bits
        if m % 3 == 0:
            return int((37*3**((m  )//3)-27)//18)
        if m % 3 == 1:
            return int((53*3**((m-7)//3)- 3)// 2)
        if m % 3 == 2:
            return int((77*3**((m-8)//3)- 3)// 2)

    @staticmethod
    def recursive(m, d, mem_calc, k=None):
        if d == 1:
            raise NotImplemented('inf')
        if d == 2:
            return UniverseSizeCalculator.d2_construction(m_bits=m, k=k)
        if d == 3 and k is None:
            return UniverseSizeCalculator.d3_construction(m_bits=m)

        start = m
        if k is not None:
            start -= (k-1)

        for n in itertools.count(start):
            if mem_calc.recursive(n, d, k) > m:
                return n - 1

    def egh(self, m_bits, d):
        """d here is for ZFD"""
        n = sum(1 for _ in itertools.takewhile(
            lambda s: s <= m_bits,
            itertools.accumulate(iter(self.pg))
        ))
        prime_prod = itertools.accumulate(
            itertools.islice(iter(self.pg), 0, n),
            operator.mul
        )
        return int(next(tail(prime_prod)) ** (1 / d))

    @staticmethod
    def ols(m_bits, d):
        """d here is for ZFD"""
        return int((m_bits/(d+1))**2)

    @staticmethod
    def ex_hamming(m_bits):
        """d=4 here is for min stopping set (3 for PD)"""
        hm = (m_bits+1)//2  # hamming parameter
        return 2**hm

    def array_ldpc(self, m_bits, s=4):
        """
        s here is for min stopping set (d=s-1 for PD)
        Construction from "More on the Stopping and Minimum Distances of Array Codes":
        s(H(2,q))=4 - mq*q^2 <- q odd prime
        s(H(4,q))>=8 - mq*q^2 <- q odd prime
        """
        start_from_ith = 1  # iterate only odd primes
        if s > 8:
            raise RuntimeError("d>8, can't assure :(")
        elif s > 6:
            # q must be at least 5
            start_from_ith = 2  # 2, 3, 5, ...
            m = 4
        elif s > 4:
            m = 3
        else:  # case for s == 4:
            m = 2

        pg = itertools.islice(iter(self.pg), start_from_ith, None)
        q = next(tail(itertools.takewhile(lambda p: p * m <= m_bits, pg)), 0)

        return q**2

    @staticmethod
    def steiner(m, d, df):
        return df[(df.d >= d) & (df.m <= m)].max().n


@attr.s
class MemoryCalculator:
    pg = attr.ib(factory=primes.Generator)
    uc = attr.ib(factory=UniverseSizeCalculator)
    rec_cache = attr.ib(factory=dict)

    @staticmethod
    def d2_construction(n):
        return math.ceil(math.log2(n + 1))

    def d3_construction(self, n):
        """d=3 here is for PD"""
        m, _ = next(filter(
            lambda t: t[1] >= n,
            enumerate(map(self.uc.d3_construction, itertools.count()))
        ))
        return m

    @staticmethod
    @cache
    def d2_k_construction(n, k):
        m = k
        while math.comb(m, k) < n:
            m += 1
        return m

    def egh(self, n, d, calc_k=False):
        """d here is for ZFD"""
        result_iterator = itertools.accumulate(iter(self.pg)) if not calc_k else itertools.count(1)
        s, _ = first_true(
            pred=lambda sp: sp[1]**(1/d) >= n,
            iterable=zip(
                result_iterator,
                itertools.accumulate(iter(self.pg), operator.mul)
            )
        )
        return s

    def recursive(self, n, d, k=None):
        if k is None:
            return self.recursive_ex(n, d)[0]
        else:
            return self.recursive_k_ex(n, d, k)[0]

    def recursive_ex(self, n, d):
        if res := self.rec_cache.get((n, d)):
            return res

        d = min(d, n)

        if d == 1 or n == 1:
            return 1, 0
        if d == 2:
            return self.d2_construction(n), 0
        if d == 3:
            return self.d3_construction(n), 0

        def m(n, d, i):
            ni = math.ceil(n / i)
            major = self.recursive_ex(ni, d)[0]
            minor = self.recursive_ex(ni, d // 2)[0]
            return major + i * minor

        res = min(
            (n, 1),
            min((m(n, d, i), i) for i in range(2, n + 1))
        )
        self.rec_cache[(n, d)] = res

        return res

    def recursive_k_ex(self, n, d, k):
        if res := self.rec_cache.get((n, d, k)):
            return res

        d = min(d, n)

        if k == 1:
            return n, 0, 0
        if d == 1:
            return k, 0, 0
        if d == 2:
            return self.d2_k_construction(n, k), 0, 0

        def m(n, d, k, i, j):
            ni = math.ceil(n / i)
            major = self.recursive_k_ex(ni, d   , k-j)[0]
            minor = self.recursive_k_ex(ni, d//2, j  )[0]
            return major + i * minor

        res = min(
            (n + k - 1, 1, 1),
            min((m(n, d, k, i, j), i, j) for i in range(2, n + 1) for j in range(1, k))
        )
        self.rec_cache[(n, d, k)] = res

        return res

    @staticmethod
    def ols(n, d):
        """d here is for ZFD"""
        return int((d + 1) * math.ceil(math.sqrt(n)))

    @staticmethod
    def ex_hamming(n):
        """d=4 here is for min stopping set (3 for PD)"""
        hm = math.ceil(math.log2(n))  # hamming parameter
        return 2 * hm - 1

    def array_ldpc(self, n, s=4):
        """
        s here is for min stopping set (d=s-1 for PD)
        Construction from "More on the Stopping and Minimum Distances of Array Codes":
        s(H(2,q))=4 - mq*q^2 <- q odd prime
        s(H(4,q))>=8 - mq*q^2 <- q odd prime
        """
        m, _ = next(filter(lambda t: t[1] >= n,
                    enumerate(map(self.uc.array_ldpc, itertools.count()))))
        return m

    def bch(self, n):
        """
        d = 4
        """
        return math.ceil(math.log2(n+1))*4

    @staticmethod
    def steiner(n, d, df):
        return df[(df.d >= d) & (df.n >= n)].m.min()


def calc_recursive_memory_job(param_list, cache_path: Path, out_dir: Path):
    cache = dict()
    if cache_path.exists():
        print('cache loaded!')
        cache = read_from_disk(cache_path)

    len_before = len(cache)
    out_path = out_dir / f'{getpid()}_{process_time_ns()}{cache_path.suffix}'

    mc = MemoryCalculator(rec_cache=cache)
    for p in param_list:
        mc.recursive(*p)

    write_to_disk(out_path, cache)

    if len_before < len(cache):
        return out_path

    return None

def try_once(rng, iblts, w, e):
    # choose erronous bits
    idxs = rng.choice(b, e)
    # words
    idxs = np.unique(idxs//w)

    # random the words themselvs
    words = rng.integers(2**w, size=len(idxs))
    # add errors to them
    error = words ^ (rng.integers(2**w-1, size=len(idxs))+1)

    # bitwise concat
    delta = (np.vstack([words, error]) + (idxs << w)).ravel()

    # add all to the filters
    return [iblt(delta) for iblt in iblts]


@attr.s
class MatrixGenerator:
    uc: UniverseSizeCalculator = attr.ib()
    mc: MemoryCalculator       = attr.ib()
    pg = attr.ib(factory=primes.Generator)
    rotate: bool = attr.ib(default=False)
    _rec_cache = attr.ib(factory=dict)

    def d2(self, n):
        cache_loc = (n, 2)
        if (res := self._rec_cache.get(cache_loc)) is not None:
            return res

        m = self.mc.d2_construction(n)

        res = np.array(list(
            itertools.islice(itertools.product((0, 1), repeat=m), 1, n + 1)
        ), dtype=MatrixAux.type)

        self._rec_cache[cache_loc] = res
        return res

    def d2_k(self, n, k):
        cache_loc = (n, 2, k)
        if (res := self._rec_cache.get(cache_loc)) is not None:
            return res

        m = self.mc.d2_k_construction(n, k)
        res = MatrixAux.F0(n, m)
        np.put_along_axis(
            res,
            indices=np.array(take(n, itertools.combinations(range(m), k)), dtype=MatrixAux.type),
            values=1,
            axis=1
        )

        self._rec_cache[cache_loc] = res
        return res

    @staticmethod
    @cache
    def recursive_d3(m):
        # TODO: use our cache
        mx = MatrixAux

        if m < 3:
            raise ValueError(f'm must be >=3, got {m}')
        if m == 3:
            return np.vstack((mx.I(3), mx.C1(3).T))
        if m == 4:
            return np.block([
                [mx.I(3),                         mx.C0(3)],
                [MatrixGenerator.recursive_d3(3), mx.C1(4)],
            ])

        i = 2 if m < 9 else 3

        h_p = MatrixGenerator.recursive_d3(m - i)
        u   = np.hstack([mx.I(3), np.zeros((3, m - 3), dtype=MatrixAux.type)])
        dl  = np.tile(h_p, (i, 1))
        dr  = np.repeat(mx.I(i), repeats=h_p.shape[0], axis=0)
        d   = np.hstack([dl, dr])
        return np.vstack([u, d])

    @staticmethod
    @cache
    def recursive_d3_weights(m):
        if m < 3:
            raise ValueError(f'm must be >=3, got {m}')
        if m == 3:
            return Counter({1: 3, 3: 1})
        if m == 4:
            return Counter({1: 3, 2: 3, 4: 1})

        i = 2 if m < 9 else 3

        c_p = MatrixGenerator.recursive_d3_weights(m - i)
        res = Counter({v + 1: c * i for v, c in c_p.items()})
        res[1] += 3
        return res

    def recursive(self, n, d, k=None):
        if k is not None:
            return self.recursive_k(n, d, k)

        cache_loc = (n, d)
        if (res := self._rec_cache.get(cache_loc)) is not None:
            return res

        mx = MatrixAux

        m, i = self.mc.recursive_ex(n, d)
        d = min(d, n)  # TODO: where this shit should sit?

        if i == 1:
            return mx.I(n)

        if d == 1 or n == 1:
            return mx.I(1)
        if d == 2:
            return self.d2(n)
        if d == 3:
            return self.recursive_d3(m)

        if i == 0:
            raise ValueError(f'{n=}, {d=}, {m=}, {i=}')

        ni = math.ceil(n / i)
        major = self.recursive(ni, d)[:ni]
        minor = self.recursive(ni, d // 2)[:ni]

        res = self._recursive_from_subs(major, minor, i)[:n]

        self._rec_cache[cache_loc] = res
        return res

    def recursive_k(self, n, d, k):
        cache_loc = (n, d, k)
        if (res := self._rec_cache.get(cache_loc)) is not None:
            return res

        mx = MatrixAux
        m, i, j = self.mc.recursive_k_ex(n, d, k)

        d = min(d, n)

        if i == 1:
            return np.hstack([mx.I(n), mx.F1(n, k-1)])

        if k == 1:
            return mx.I(n)
        if d == 1:
            return mx.F1(n, k)
        if d == 2:
            return self.d2_k(n, k)

        ni = math.ceil(n / i)
        major = self.recursive_k(ni, d   , k-j)[:ni]
        minor = self.recursive_k(ni, d//2, j  )[:ni]

        res = self._recursive_from_subs(major, minor, i)[:n]

        self._rec_cache[cache_loc] = res
        return res

    def _recursive_from_subs(self, major, minor, i):
        if self.rotate:
            majors = np.vstack([np.roll(major, j*major.shape[1]//i, 1) for j in range(i)])
        else:
            majors = np.tile(major, (i, 1))
        minors = np.vstack([
            minor, np.tile(np.zeros_like(minor), (i - 1, 1))
        ])
        minors = np.hstack([np.roll(minors, j * len(minor), axis=0) for j in range(i)])
        return np.hstack([majors, minors])

    def egh(self, m, d, n=None):
        if n is None:
            n = self.uc.egh(m, d)

        def level(p):
            return np.tile(MatrixAux.I(p), math.ceil(n / p))[:, :n].T

        levels = [
            level(p) for p, _ in
            itertools.takewhile(lambda ps: ps[1] <= m,
                                zip(self.pg, itertools.accumulate(self.pg)))
        ]
        return np.hstack(levels)

    @staticmethod
    def ols(s, d):
        b1 = np.vstack([np.ones(s, dtype=MatrixAux.type), np.zeros((s - 1, s), dtype=MatrixAux.type)])
        bi = np.eye(s, dtype=MatrixAux.type)

        first_row = np.hstack([np.roll(b1, i, axis=0) for i in range(s)])
        bottom = np.vstack([np.hstack([np.roll(bi, (r * i) % s, axis=0) for i in range(s)]) for r in range(d)])
        return np.vstack([first_row, bottom]).T

    @staticmethod
    def iblt(m, n, k=4):
        bucket_size = m / k

        a = np.zeros((n, m), dtype=MatrixAux.type)

        for i in range(n):
            for j in range(k):
                b0 = round(bucket_size * j)
                b1 = round(bucket_size * (j + 1))
                a[i, b0 + int(mmh3(repr(j), i) % (b1 - b0))] = 1
        return a

    @staticmethod
    def random_iblt(m, n, k=4):
        rng = np.random.default_rng()
        if m < 750:  # time optimization
            a = np.hstack([np.zeros((n, m - k)), np.ones((n, k))]).astype(MatrixAux.type)
            for r in a:
                np.random.shuffle(r)
        else:
            a = np.zeros((n, m), dtype=MatrixAux.type)
            for r in a:
                r[rng.choice(m, k, replace=False)] = 1

        return a


class ColumnGenerator(ABC):
    @abstractmethod
    def iterable(self, key: int) -> Iterable:
        pass

    @abstractmethod
    def list(self, key: int) -> list:
        pass

    @abstractmethod
    def array(self, key: int, a: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def size_bits(self) -> int:
        pass


@dataclass
class ColumnGeneratorEGH(ColumnGenerator):
    m: int
    pg: primes.Generator

    def size_bits(self) -> int:
        # make sure the generator is initialized with the relevant primes
        self.list(0)
        return len(self.pg.primes) * math.ceil(math.log2(self.m))

    def iterable(self, key: int) -> Iterable:
        it = map(sum,
                 ((key % p, s) for p, s in itertools.takewhile(
                     lambda ps: ps[1] <= self.m,
                     zip(self.pg, itertools.chain([0], itertools.accumulate(self.pg)))
                 )))
        prev = next(it, None)
        if prev is None:
            return
        for current in it:
            yield prev
            prev = current

    def list(self, key: int) -> list:
        return list(self.iterable(key))

    def array(self, key: int, a: np.ndarray) -> np.ndarray:
        prime_sum = 0
        for p in self.pg:
            if prime_sum + p > self.m:
                break
            a[prime_sum + (key % p)] = 1
            prime_sum += p
        return a


class ColumnGeneratorD3(ColumnGenerator):
    def __init__(self, n, mc: MemoryCalculator):
        m = mc.d3_construction(n)
        if m < 3:
            raise ValueError(f'm must be >=3, got {m}')

        self.m = m
        self.n = n
        i = 0 if m<4 else 1 if m==4 else 2 if m<9 else 3

        if i:
            self.ni = math.ceil((self.n-3) / i)
            self.ni_calc = type(self)(self.ni, mc)

    def size_bits(self) -> int:
        return 2 * math.ceil(math.log2(self.m))

    def iterable(self, key: int) -> Iterable:
        if key < 3:
            # first block is only with one I
            yield key
            return

        if self.m < 4:
            # last column of base case
            yield from range(3)
            return

        # now for the main matrix
        key -= 3

        yield from self.ni_calc.iterable(key % self.ni)
        i_block = key // self.ni
        yield self.m - i_block - 1

    def list(self, key: int) -> list:
        if key < 3:
            # first block is only with one I
            return [key]

        if self.m < 4:
            # last column of base case
            return list(range(3))

        # now for the main matrix
        key -= 3

        idxs = self.ni_calc.list(key % self.ni)
        i_block = key // self.ni
        idxs.append(self.m - i_block - 1)
        return idxs

    def array(self, key: int, a: np.ndarray):
        if key < 3:
            # first block is only with one I
            a[key] = 1

        elif self.m < 4:
            # last column of base case
            a[:3] = 1

        else:
            # now for the main matrix
            key -= 3

            self.ni_calc.array(key % self.ni, a)
            i_block = key // self.ni
            a[self.m - i_block - 1] = 1

        return a


class ColumnGeneratorHamming(ColumnGenerator):
    def __init__(self, m, extended=True):
        if m < 3:
            raise ValueError(f'm must be >=3, got {m}')

        self.mask = 2**np.arange(m)
        self.ex = extended

    def size_bits(self) -> int:
        return math.ceil(math.log2(len(self.mask)))

    def iterable(self, key: int) -> Iterable:
        x = key
        i = 0
        while x != 0:
            x2 = x//2
            if x - x2*2 == 1:
                yield i
            x = x2
            i += 1

        if self.ex:
            yield len(self.mask)-1

    def list(self, key: int) -> list:
        if self.ex:
            return [*(key & self.mask).nonzero()[0], len(self.mask)-1]
        return list((key & self.mask).nonzero()[0])

    def array(self, key: int, a: np.ndarray):
        np.not_equal(0, (key & self.mask), out=a)
        if self.ex:
            a[-1] = 1
        return a


class ColumnGeneratorK(ColumnGenerator):
    def __init__(self, n, d, k, mc: MemoryCalculator):
        m, i, j = mc.recursive_k_ex(n, d, k)
        self.d = d
        self.k = k
        self.m = m
        self.i = i

        if self._has_children():
            self.ni = math.ceil(n / i)
            self.major = type(self)(self.ni, d   , k-j, mc)
            self.minor = type(self)(self.ni, d//2, j  , mc)

    def _has_children(self):
        return self.k > 1 and self.d > 2 and self.i > 1

    def size_bits(self) -> int:
        size = 4 * math.ceil(math.log2(self.m))
        if self._has_children():
            size += self.major.size_bits()
            size += self.minor.size_bits()
        return size

    def write_combination(self, i, a):
        remaining_ones = self.k

        for j in range(self.m):
            max_index = self.m - j - 1
            combinations_without_max = math.comb(max_index, remaining_ones)

            if i >= combinations_without_max:
                a[j] = 1
                i -= combinations_without_max
                remaining_ones -= 1

        return a

    def iter_combination(self, i):
        remaining_ones = self.k

        for j in range(self.m):
            max_index = self.m - j - 1
            combinations_without_max = math.comb(max_index, remaining_ones)

            if i >= combinations_without_max:
                yield j
                i -= combinations_without_max
                remaining_ones -= 1

    def iterable(self, key: int) -> Iterable:
        if self.i == 1:
            yield key
            yield from range(self.m - self.k + 1, self.m)
        elif self.k == 1:
            yield key
        elif self.d == 1:
            yield from range(self.k)
        elif self.d == 2:
            yield from self.iter_combination(key)
        else:
            # now for the main matrix
            i_block = key // self.ni
            minor_start = self.major.m + i_block*self.minor.m
            yield from self.major.list(key % self.ni)
            for idx in self.minor.list(key % self.ni):
                yield idx + minor_start

    def list(self, key: int) -> list:
        if self.i == 1:
            return [key] + list(range(self.m - self.k + 1, self.m))
        if self.k == 1:
            return [key]
        if self.d == 1:
            return list(range(self.k))
        if self.d == 2:
            return list(self.iter_combination(key))

        # now for the main matrix
        i_block = key // self.ni
        minor_start = self.major.m + i_block*self.minor.m
        res = self.major.list(key % self.ni)
        res += [idx + minor_start for idx in self.minor.list(key % self.ni)]
        return res

    def array(self, key: int, a: np.ndarray):
        if self.i == 1:
            a[key] = 1
            a[-self.k+1:] = 1
        elif self.k == 1:
            a[key] = 1
        elif self.d == 1:
            a[:self.k] = 1
        elif self.d == 2:
            self.write_combination(key, a)
        else:
            # now for the main matrix
            i_block = key // self.ni
            minor_start = self.major.m + i_block*self.minor.m
            self.major.array(key % self.ni, a.view()[:self.major.m])
            self.minor.array(key % self.ni, a.view()[minor_start:minor_start+self.minor.m])
        return a


class ColumnGeneratorRec(ColumnGenerator):
    def __init__(self, n, d, mc: MemoryCalculator):
        m, i = mc.recursive_ex(n, d)
        self.d = d
        self.m = m
        self.i = i

        gen = None
        if d == 2:
            gen = ColumnGeneratorHamming(m, False)
        if d == 3:
            gen = ColumnGeneratorD3(n, mc)
        if gen:
            self.iterable = gen.iterable
            self.list = gen.list
            self.array = gen.array
            self.size_bits = gen.size_bits
            return

        if d > 3 and i > 1:
            self.ni = math.ceil(n / i)
            self.major = type(self)(self.ni, d   , mc)
            self.minor = type(self)(self.ni, d//2, mc)

    def size_bits(self) -> int:
        size = 3 * math.ceil(math.log2(self.m))
        if self.i > 1:
            size += self.major.size_bits()
            size += self.minor.size_bits()
        return size

    def iterable(self, key: int) -> Iterable:
        if self.i == 1:
            yield key
        else:
            # now for the main matrix
            i_block = key // self.ni
            minor_start = self.major.m + i_block*self.minor.m
            yield from self.major.list(key % self.ni)
            for idx in self.minor.list(key % self.ni):
                yield idx + minor_start

    def list(self, key: int) -> list:
        if self.i == 1:
            return [key]

        # now for the main matrix
        i_block = key // self.ni
        minor_start = self.major.m + i_block*self.minor.m
        res = self.major.list(key % self.ni)
        res += [idx + minor_start for idx in self.minor.list(key % self.ni)]
        return res

    def array(self, key: int, a: np.ndarray):
        if self.i == 1:
            a[key] = 1
        else:
            # now for the main matrix
            i_block = key // self.ni
            minor_start = self.major.m + i_block*self.minor.m
            self.major.array(key % self.ni, a.view()[:self.major.m])
            self.minor.array(key % self.ni, a.view()[minor_start:minor_start+self.minor.m])
        return a


@dataclass
class HashMapper(ColumnGenerator):
    m: int
    k: int = 3
    base_hash: Any = mmh3

    def size_bits(self) -> int:
        return (self.k + 1) * math.ceil(math.log2(self.m))

    def hash(self, i, key):
        base = abs(self.base_hash(str(key), i))
        in_k_sub = base % (self.m // self.k)
        index_in_table = in_k_sub + (self.m // self.k) * i
        return index_in_table

    def iterable(self, key: int) -> Iterable:
        return (self.hash(i, key) for i in range(self.k))

    def list(self, key: int) -> list:
        return [self.hash(i, key) for i in range(self.k)]

    def array(self, key: int, a: np.ndarray) -> np.ndarray:
        for i in range(self.k):
            a[self.hash(i, key)] = 1
        return a


class MatrixIndicesMapper(ColumnGenerator):
    def __init__(self, M: np.ndarray, m=None):
        self.M = M.astype(MatrixAux.type)
        if m:
            if self.M.shape[1] < m:
                self.M = np.pad(self.M, [(0,0),(0,m-self.M.shape[1])], constant_values=0)
            if self.M.shape[1] > m:
                self.M = self.M[:,:m]

    def size_bits(self) -> int:
        return self.M.size()

    def iterable(self, key: int) -> Iterable:
        return self.list(key)

    def list(self, key: int) -> list:
        return self.M[key].nonzero()[0]

    def array(self, key: int, a: np.ndarray) -> np.ndarray:
        return self.M[key]


class SparseFixedMatrixIndicesMapper(ColumnGenerator):
    def __init__(self, iM):
        self.iM: np.ndarray = iM

    def size_bits(self) -> int:
        return 8 * self.iM.nbytes

    @staticmethod
    def from_full_matrix(M):
        w = M.sum(1).max()
        iM = np.empty((len(M), w), dtype=np.uint16)

        for i, e in enumerate(M):
            iM[i] = e.nonzero()[0]

        return SparseFixedMatrixIndicesMapper(iM)

    @staticmethod
    def from_column_generator(n, k, column_generator: ColumnGenerator):
        iM = np.empty((n, k), dtype=np.uint16)

        for i in range(n):
            iM[i] = column_generator.list(i)

        return SparseFixedMatrixIndicesMapper(iM)

    def iterable(self, key: int) -> Iterable:
        return self.iM[key]

    def list(self, key: int) -> list:
        return list(self.iM[key])

    def array(self, key: int, a: np.ndarray) -> np.ndarray:
        a[self.iM[key]] = 1
        return a


class SparseIndicesMapper(ColumnGenerator):
    def __init__(self, n, column_generator: ColumnGenerator):
        self.map = list(map(column_generator.list, range(n)))

    def size_bits(self) -> int:
        size = 32
        for row in self.map:
            size += len(row) * 16 + 32

        return size

    def iterable(self, key: int) -> Iterable:
        yield from self.map[key]

    def list(self, key: int) -> list:
        return self.map[key]

    def array(self, key: int, a: np.ndarray) -> np.ndarray:
        a[self.map[key]] = 1
        return a


def calc_space_rec(n, d, workdir: Path):
    id_str = f'REC{d}'
    pg = primes.Generator()
    uc = UniverseSizeCalculator(pg)
    mc = MemoryCalculator(pg, uc)
    mc.rec_cache.update(read_from_disk('recursive_dict.pkl'))

    m = mc.recursive(n=n, d=d)
    mapper = ColumnGeneratorRec(n=n, d=d, mc=mc)

    expected_matrix_size_bytes = m * n
    sparse = SparseIndicesMapper(n, mapper).size_bits() if expected_matrix_size_bytes <= 2**31 else np.NaN

    result = {
        'n': n, 'm': m, 'alg': id_str,
        'M': m*n,
        'S': sparse,
        'C': mapper.size_bits(),
    }

    pd.DataFrame([result]).to_csv(workdir / f'{n}{id_str}_{process_time_ns()}.csv', index=False)


def calc_space_rec_k(n, d, k, workdir: Path):
    id_str = f'REC{d}k{k}'
    pg = primes.Generator()
    uc = UniverseSizeCalculator(pg)
    mc = MemoryCalculator(pg, uc)
    mc.rec_cache.update(read_from_disk('recursive_dict.pkl'))

    m = mc.recursive(n=n, d=d, k=k)
    mapper = ColumnGeneratorK(n=n, d=d, k=k, mc=mc)

    result = {
        'n': n, 'm': m, 'alg': id_str,
        'M': m*n,
        'S': n*k*16,
        'C': mapper.size_bits(),
    }

    pd.DataFrame([result]).to_csv(workdir / f'{n}{id_str}_{process_time_ns()}.csv', index=False)


def calc_space_egh(n, d, workdir: Path):
    id_str = f'EGH{d}'
    pg = primes.Generator()
    uc = UniverseSizeCalculator(pg)
    mc = MemoryCalculator(pg, uc)

    m = mc.egh(n=n, d=d-1)
    k = mc.egh(n=n, d=d-1, calc_k=True)

    mapper = ColumnGeneratorEGH(m=m, pg=pg)

    result = {
        'n': n, 'm': m, 'alg': id_str,
        'M': m*n,
        'S': n*k*2*8,
        'C': mapper.size_bits(),
    }

    pd.DataFrame([result]).to_csv(workdir / f'{n}{id_str}_{process_time_ns()}.csv', index=False)


def insertion_from_list(count, key_x, val_x, e, column_generator: ColumnGenerator):
    for i in column_generator.list(e):
        count[i] += 1
        key_x[i] ^= e
        val_x[i] ^= (e*13 - 11) ^ 31


def insertion_from_iterable(count, key_x, val_x, e, column_generator: ColumnGenerator):
    for i in column_generator.iterable(e):
        count[i] += 1
        key_x[i] ^= e
        val_x[i] ^= (e*13 - 11) ^ 31


def insertion_from_array(count, key_x, val_x, e, column_generator: ColumnGenerator):
    a = np.zeros_like(count)
    a = column_generator.array(e, a)
    count += a
    key_x ^= a * e
    val_x ^= a * ((e * 13 - 11) ^ 31)


insertion_methods = {
    'a': insertion_from_array,
    'l': insertion_from_list,
    'i': insertion_from_iterable,
}


def measure_insertions(m: int, column_generator: ColumnGenerator, elements: Iterable, timer, inserter, dtype=np.int32):
    count = np.zeros(m, dtype=dtype)
    key_x = np.zeros(m, dtype=dtype)
    val_x = np.zeros(m, dtype=dtype)

    total_time = 0
    for e in elements:
        t0 = timer()
        inserter(count, key_x, val_x, e, column_generator)
        total_time += timer() - t0
    return total_time


def measure_column_generation(column_generator: ColumnGenerator, elements: Iterable, timer):
    total_time = 0
    for e in elements:
        t0 = timer()
        column_generator.list(e)
        total_time += timer() - t0
    return total_time


def universe_generator(n, sample_size):
    return (random.randrange(n) for _ in range(sample_size))


def generate_logger(id_str, n):
    def logger(s):
        print(f'{id_str}, {n =:3}', s)
    return logger


# TODO fix name
def measure_return_df(id_str, mapper: ColumnGenerator, m, n, workdir, reps, sample_size):
    logger = generate_logger(id_str, n)
    mappers = {
        id_str: mapper,
        'IBLTmmh': HashMapper(m=m),
    }
    for rep in range(reps):
        logger(f'[{rep + 1}/{reps}]')
        result = {'n': n, 'm': m, 'rep': rep}

        for name, mapper in mappers.items():
            for char, insertion_method in insertion_methods.items():
                result[f'{name}[{char}]'] = measure_insertions(
                    m,
                    mapper,
                    universe_generator(n, sample_size),
                    process_time_ns,
                    insertion_from_array
                )

        pd.DataFrame([result]).to_csv(workdir / f'{n}{id_str}_{process_time_ns()}.csv', index=False)


def measure_generation_clean(id_str, mapper: ColumnGenerator, n, workdir, reps, sample_size):
    logger = generate_logger(id_str, n)
    for rep in range(reps):
        logger(f'[{rep + 1}/{reps}]')
        result = {'n': n, 'rep': rep}

        result[id_str] = measure_column_generation(
            mapper,
            universe_generator(n, sample_size),
            process_time_ns,
        )

        pd.DataFrame([result]).to_csv(workdir / f'{n}{id_str}_{process_time_ns()}.csv', index=False)


def measure_d3_column_generator(n, workdir: Path, reps=10, sample_size=10**6):
    id_str = 'C_d3'
    logger = generate_logger(id_str, n)

    pg = primes.Generator()
    uc = UniverseSizeCalculator(pg)
    mc = MemoryCalculator(pg, uc)

    logger('computing m')
    m = mc.d3_construction(n=n)
    logger(f'{m=}')

    logger('creating ColumnGenerator')
    mapper = ColumnGeneratorD3(n=n, mc=mc)

    return measure_return_df(id_str, mapper, m, n, workdir, reps, sample_size)


def measure_k_column_generator(n, d, k, workdir: Path, reps=10, sample_size=10**6):
    id_str = f'C_d{d}k{k}'
    logger = generate_logger(id_str, n)

    pg = primes.Generator()
    uc = UniverseSizeCalculator(pg)
    mc = MemoryCalculator(pg, uc)

    logger('updating cache')
    mc.rec_cache.update(read_from_disk('recursive_dict.pkl'))

    logger('computing m')
    m = mc.recursive(n=n, d=d, k=k)
    logger(f'{m=}')

    logger('creating ColumnGenerator')
    mapper = ColumnGeneratorK(n, d, k, mc)

    return measure_return_df(id_str, mapper, m, n, workdir, reps, sample_size)


def measure_k_column_generator_clean(n, d, k, workdir: Path, reps=10, sample_size=10**6):
    id_str = f'C_d{d}k{k}'
    logger = generate_logger(id_str, n)

    pg = primes.Generator()
    uc = UniverseSizeCalculator(pg)
    mc = MemoryCalculator(pg, uc)

    logger('updating cache')
    mc.rec_cache.update(read_from_disk('recursive_dict.pkl'))

    logger('creating ColumnGenerator')
    mapper = ColumnGeneratorK(n, d, k, mc)

    return measure_generation_clean(id_str, mapper, n, workdir, reps, sample_size)


def measure_rec_column_generator(n, d, workdir: Path, reps=10, sample_size=10**6):
    id_str = f'C_d{d}'
    logger = generate_logger(id_str, n)

    pg = primes.Generator()
    uc = UniverseSizeCalculator(pg)
    mc = MemoryCalculator(pg, uc)

    logger('updating cache')
    mc.rec_cache.update(read_from_disk('recursive_dict.pkl'))

    logger('computing m')
    m = mc.recursive(n=n, d=d)
    logger(f'{m=}')

    logger('creating ColumnGenerator')
    mapper = ColumnGeneratorRec(n, d, mc)

    return measure_return_df(id_str, mapper, m, n, workdir, reps, sample_size)


def measure_rec_column_generator_clean(n, d, workdir: Path, reps=10, sample_size=10**6):
    id_str = f'C_d{d}'
    logger = generate_logger(id_str, n)

    pg = primes.Generator()
    uc = UniverseSizeCalculator(pg)
    mc = MemoryCalculator(pg, uc)

    logger('updating cache')
    mc.rec_cache.update(read_from_disk('recursive_dict.pkl'))

    logger('creating ColumnGenerator')
    mapper = ColumnGeneratorRec(n, d, mc)

    return measure_generation_clean(id_str, mapper, n, workdir, reps, sample_size)


def measure_k_matrix(n, d, k, workdir: Path, reps=10, sample_size=10**6):
    id_str = f'M_d{d}k{k}'
    logger = generate_logger(id_str, n)

    pg = primes.Generator()
    uc = UniverseSizeCalculator(pg)
    mc = MemoryCalculator(pg, uc)
    matgen = MatrixGenerator(uc, mc, pg)

    logger('updating cache')
    mc.rec_cache.update(read_from_disk('recursive_dict.pkl'))

    logger('computing m')
    m = mc.recursive(n=n, d=d, k=k)
    logger(f'{m=}')

    expected_matrix_size_bytes = m * n
    if expected_matrix_size_bytes > 2**31:
        logger(f'Matrix is too large: {(expected_matrix_size_bytes/2**30):.2}GiB')
        return None

    logger('creating ColumnGenerator')
    mapper = MatrixIndicesMapper(matgen.recursive(n=n, d=d, k=k), m=m)

    return measure_return_df(id_str, mapper, m, n, workdir, reps, sample_size)


def measure_rec_matrix(n, d, workdir: Path, reps=10, sample_size=10**6):
    id_str = f'M_d{d}'
    logger = generate_logger(id_str, n)

    pg = primes.Generator()
    uc = UniverseSizeCalculator(pg)
    mc = MemoryCalculator(pg, uc)
    matgen = MatrixGenerator(uc, mc, pg)

    logger('updating cache')
    mc.rec_cache.update(read_from_disk('recursive_dict.pkl'))

    logger('computing m')
    m = mc.recursive(n=n, d=d)
    logger(f'{m=}')

    expected_matrix_size_bytes = m * n
    if expected_matrix_size_bytes > 2**31:
        logger(f'Matrix is too large: {(expected_matrix_size_bytes/2**30):.2}GiB')
        return None

    logger('creating ColumnGenerator')
    mapper = MatrixIndicesMapper(matgen.recursive(n=n, d=d), m=m)

    return measure_return_df(id_str, mapper, m, n, workdir, reps, sample_size)


def measure_k_sparse_matrix(n, d, k, workdir: Path, reps=10, sample_size=10**6):
    id_str = f'S_d{d}k{k}'
    logger = generate_logger(id_str, n)

    pg = primes.Generator()
    uc = UniverseSizeCalculator(pg)
    mc = MemoryCalculator(pg, uc)

    logger('updating cache')
    mc.rec_cache.update(read_from_disk('recursive_dict.pkl'))

    logger('computing m')
    m = mc.recursive(n=n, d=d, k=k)
    logger(f'{m=}')

    expected_matrix_size_bytes = k * n * 2
    if expected_matrix_size_bytes > 2**31:
        logger(f'Matrix is too large: {(expected_matrix_size_bytes/2**30):.2}GiB')
        return None

    logger('creating ColumnGenerator')
    mapper = SparseFixedMatrixIndicesMapper.from_column_generator(n, k, ColumnGeneratorK(n, d, k, mc))

    return measure_return_df(id_str, mapper, m, n, workdir, reps, sample_size)


def measure_rec_sparse_matrix(n, d, workdir: Path, reps=10, sample_size=10**6):
    id_str = f'S_d{d}'
    logger = generate_logger(id_str, n)

    pg = primes.Generator()
    uc = UniverseSizeCalculator(pg)
    mc = MemoryCalculator(pg, uc)

    logger('updating cache')
    mc.rec_cache.update(read_from_disk('recursive_dict.pkl'))

    logger('computing m')
    m = mc.recursive(n=n, d=d)
    logger(f'{m=}')

    expected_matrix_size_bytes = n * 2 * (n//3)
    if expected_matrix_size_bytes > 2**31:
        logger(f'Matrix is too large: {(expected_matrix_size_bytes/2**30):.2}GiB')
        return None

    logger('creating ColumnGenerator')
    mapper = SparseIndicesMapper(n, ColumnGeneratorRec(n, d, mc))

    return measure_return_df(id_str, mapper, m, n, workdir, reps, sample_size)


def measure_egh_column_generator(n, d, workdir: Path, reps=10, sample_size=10**6):
    id_str = f'C_EGH{d}'
    logger = generate_logger(id_str, n)

    pg = primes.Generator()
    uc = UniverseSizeCalculator(pg)
    mc = MemoryCalculator(pg, uc)

    logger('computing m')
    m = mc.egh(n=n, d=d-1)
    logger(f'{m=}')

    logger('creating ColumnGenerator')
    mapper = ColumnGeneratorEGH(m=m, pg=pg)

    return measure_return_df(id_str, mapper, m, n, workdir, reps, sample_size)


def measure_egh_column_generator_clean(n, d, workdir: Path, reps=10, sample_size=10**6):
    id_str = f'C_EGH{d}'
    logger = generate_logger(id_str, n)

    pg = primes.Generator()
    uc = UniverseSizeCalculator(pg)
    mc = MemoryCalculator(pg, uc)

    logger('computing m')
    m = mc.egh(n=n, d=d-1)
    logger(f'{m=}')

    logger('creating ColumnGenerator')
    mapper = ColumnGeneratorEGH(m=m, pg=pg)

    return measure_generation_clean(id_str, mapper, n, workdir, reps, sample_size)


def measure_egh_matrix(n, d, workdir: Path, reps=10, sample_size=10**6):
    id_str = f'M_EGH{d}'
    logger = generate_logger(id_str, n)

    pg = primes.Generator()
    uc = UniverseSizeCalculator(pg)
    mc = MemoryCalculator(pg, uc)
    matgen = MatrixGenerator(uc, mc, pg)

    logger('computing m')
    m = mc.egh(n=n, d=d-1)
    logger(f'{m=}')

    expected_matrix_size_bytes = m * n
    if expected_matrix_size_bytes > 2**31:
        logger(f'Matrix is too large: {(expected_matrix_size_bytes/2**30):.2}GiB')
        return None

    logger('creating ColumnGenerator')
    mapper = MatrixIndicesMapper(matgen.egh(m=m, d=d, n=n), m=m)

    return measure_return_df(id_str, mapper, m, n, workdir, reps, sample_size)


def measure_egh_sparse_matrix(n, d, workdir: Path, reps=10, sample_size=10**6):
    id_str = f'S_EGH{d}'
    logger = generate_logger(id_str, n)

    pg = primes.Generator()
    uc = UniverseSizeCalculator(pg)
    mc = MemoryCalculator(pg, uc)

    logger('computing m, k')
    m = mc.egh(n=n, d=d-1)
    k = mc.egh(n=n, d=d-1, calc_k=True)
    logger(f'{m=},{k=}')

    expected_matrix_size_bytes = k * n * 2
    if expected_matrix_size_bytes > 2**31:
        logger(f'Matrix is too large: {(expected_matrix_size_bytes/2**30):.2}GiB')
        return None

    logger('creating ColumnGenerator')
    mapper = SparseFixedMatrixIndicesMapper.from_column_generator(n, k, ColumnGeneratorEGH(m=m, pg=pg))

    return measure_return_df(id_str, mapper, m, n, workdir, reps, sample_size)


def hash_collision(rng, n_items, n_buckets):
    return (np.unique(rng.integers(0, high=n_items, size=n_buckets), return_counts=True)[1]>1).any()


def hash_collision_prob(n_items, n_buckets, reps=10**5):
    rng = np.random.default_rng()
    return sum(itertools.starmap(hash_collision, itertools.repeat((rng, n_items, n_buckets), reps)))/reps


def expected_sync_size(ds_size, failure_prob, penalty_size):
    return ds_size + failure_prob*penalty_size


def expected_sync_size_lffz(n, d, phi, lffz_sizer, I=4*4, N=5):
    return expected_sync_size(
        ds_size=lffz_sizer(n=n, d=d)*I,
        failure_prob=hash_collision_prob(n_items=N, n_buckets=n),
        penalty_size=phi
    )


def old_expected_sync_size_iblt(phi, I=4*4, N=5):
    size, p = IBLT_success_rates.fetch_optimal_iblt_size_p(N)
    return expected_sync_size(
        ds_size=size*I,
        failure_prob=1-p,
        penalty_size=phi
    )


def expected_sync_size_iblt(size, p, phi, I=4*4, N=5):
    return expected_sync_size(
        ds_size=size*I,
        failure_prob=1-p,
        penalty_size=phi
    )


def inverse_selection(code, indexes):
    mask = np.ones(len(code), bool)
    mask[indexes,] = 0
    return code[mask]


def are_idxs_cover_any(code, indexes):
    assert code.dtype == bool

    y = code[indexes, :].any(0)
    others = inverse_selection(code, indexes)
    return ((others | y) == y).all(1).any()


def are_idxs_sum_has_1(code, indexes):
    assert code.dtype == bool

    return (code[indexes, :].sum(0) == 1).any()


def is_zfd(code, d):
    assert code.dtype == bool

    for indexes in itertools.combinations(range(len(code)), d):
        if are_idxs_cover_any(code, indexes):
            return False

    return True


def get_pd_stopping_set(code, d):
    return next(pd_stopping_sets_generator(code, d), None)


def pd_stopping_sets_generator(code, d):
    assert code.dtype == bool

    for indexes in itertools.combinations(range(len(code)), d):
        if not are_idxs_sum_has_1(code, indexes):
            yield indexes


def is_pd(code, d):
    return not get_pd_stopping_set(code, d)


def find_max_d(code, is_d_checker):
    d = 1
    for i in range(1, len(code) + 1):
        if not is_d_checker(code, d + 1):
            break
        d = i
    return d


@attr.s
class Result:
    d    = attr.ib(default=0)
    rank = attr.ib(default=0, cmp=False)
    data = attr.ib(default=None, cmp=False, repr=False)

    def __str__(self):
        return f'd={self.d}, rank={self.rank}\n{self.data}\n'


def get_the_best(res_list):
    return [max(*z) for z in zip(*res_list)]


def iterate_exclusive(it, i, n_parts):
    return itertools.islice(it, i, None, n_parts)


def read_from_disk(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def write_to_disk(path, obj):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


pickle_file_pattern = '{idx}.pkl'


def get_pickle_dir(n, m, k):
    return Path(f'best_fpfz_n{n}_k{k}_m{m}')


def on_success(i, bests, nmk):
    file = get_pickle_dir(*nmk) / pickle_file_pattern.format(idx=i)
    write_to_disk(file, bests)


def find_max_d_for_job(i, n_jobs, codewords, nmk):
    n, m, k = nmk
    used_codes = CodeCache()
    miss = 0

    bests = [Result(), Result()]
    max_rank = n - 1
    roof = [Result(n, max_rank), Result(n, max_rank)]

    subsets = iterate_exclusive(itertools.combinations(range(len(codewords)), n), i, n_jobs)
    count = 0

    for indexes in subsets:
        # if we hit the roof, no need to continue
        if bests == roof:
            print("Roof was hit", bests)
            break

        code = codewords[indexes, :]

        #if not used_codes.add(code):
        #    continue
        miss += 1

        rank = n - np.linalg.matrix_rank(code)
        results = [
            roof[0],
            Result(find_max_d(code, is_pd), rank, indexes),
        ]
        count += 1
        bests = get_the_best([bests, results])

    print(f'finished {count} subsets, best_d={bests}. Hit-rate={1 - miss / count:5.2%}')
    on_success(i, bests, nmk)
    return bests


@attr.s
class CodeCache:
    _set = attr.ib(factory=set)

    def add(self, code):
        """returns True if the code was inserted to the cache"""
        pre_size = len(self._set)
        self._set.add(get_code_id(code))
        post_size = len(self._set)
        return pre_size != post_size


def get_code_id(code):
    code = np.packbits(code, 0)
    return code.take(np.lexsort(code, 0), 1).tobytes()


def sort_code(code):
    return code.take(np.lexsort(code, 0), 1)


def check_futures(futures, current_best):
    not_ready = []

    for future in futures:
        if not future.ready():
            not_ready.append(future)
            continue
        if not future.successful():
            continue

        result = future.get()
        current_best = get_the_best([current_best, result])

    return not_ready, current_best


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', default=5, type=int, help='code length')
    parser.add_argument('-n', default=4, type=int, help='code size (number of codewords)')
    parser.add_argument('-k', default=3, type=int, help='codeword Hamming weight, set 0 for unrestricted weight')
    parser.add_argument('--skip-exist', action='store_true', help='return immediately if there is .txt file for n,m,k')
    parser.add_argument('--keep-dir', action='store_true', help='keep working directory after process is finished')
    parser.add_argument('--only-print-universe', action='store_true', help='print the shuffled universe and exit')
    parser.add_argument('--skip-combs-gt', type=int, help='codeword Hamming weight')

    return parser.parse_args()


def write_readable_results(output_file, codewords, bests, nmk):
    n, m, k = nmk
    with output_file.open('w') as f:
        f.write(f'n={n}, m={m}, k={k}\n')
        for i, name in enumerate(('ZFD', 'PD')):
            best = bests[i]
            f.write(f'{name}: best d={best.d}, rank={best.rank}\n')
            for row in sort_code(codewords[best.data, :]).astype(int):
                print(row, file=f)


if __name__ == '__main__':
    """
    this code is here in __main__ and not in a dedicated function,
     because I want to explore the variables here post mortem.
    """
    args = parse_args()
    print('ARGS:', args)
    n, m, k = args.n, args.m, args.k

    pickle_file_dir = get_pickle_dir(n, m, k)
    n_valid_vectors = math.comb(m, k) if k != 0 else (2**m - 1)
    n_combinations = math.comb(n_valid_vectors, n)
    print(f'n_combinations={n_combinations:,}')

    print('TODO: IGNORING ZFD, results are garbage')

    output_file = pickle_file_dir.with_suffix('.txt')
    shutil.rmtree(pickle_file_dir, ignore_errors=True)

    if args.skip_exist and output_file.exists():
        print(f'EXIT: {output_file} already exists')
        sys.exit(1)
    if args.skip_combs_gt and args.skip_combs_gt < n_combinations:
        print('EXIT: too many combs')
        sys.exit(1)

    pickle_file_dir.mkdir(parents=True, exist_ok=True)

    # generate all codewords for m,k
    if k != 0:
        codewords = set(itertools.permutations((1,) * k + (0,) * (m - k)))
    else:  # unrestricted codeword weight
        codewords = itertools.product((0, 1), repeat=m)
    codewords = np.array(list(codewords)).astype(bool)
    # remove the zero vector, if exists
    codewords = codewords[codewords.any(1)]

    np.random.seed(0)
    np.random.shuffle(codewords)  # progress in random order
    assert n_combinations == math.comb(len(codewords), n)
    if args.only_print_universe:
        print(repr(codewords.astype(int)))
        sys.exit(0)

    # distribute chunks among the CPUs, but don't make single chunk too long (cache bloat, less checkpoints)
    max_chunk_size = 10 ** 4
    max_jobs_per_cpu = 100
    max_online_jobs = cpu_count() * max_jobs_per_cpu
    chunk_size = n_combinations // cpu_count()
    chunk_size = max(1, min(chunk_size, max_chunk_size))
    n_chunks = -(n_combinations // -chunk_size)
    print(f'n_chunks,chunk_size={n_chunks}, {chunk_size}')

    bests = [Result(), Result()]
    roof = [Result(n, n-1), Result(n, n-1)]

    start = datetime.now()
    with Pool(maxtasksperchild=1) as pool:
        sleep_time = 1
        max_sleep_time = 10
        job_id_iter = iter(range(n_chunks))
        futures = []
        while True:
            new_futures = [pool.apply_async(find_max_d_for_job, (i, n_chunks, codewords, (n, m, k)))
                           for i in take(max_online_jobs - len(futures), job_id_iter)]
            futures += new_futures
            if not futures:
                break
            futures, bests = check_futures(futures, bests)
            if bests == roof:
                # no need to continue
                print("Terminating pool because we found ", bests)
                pool.terminate()
                break
            sleep(sleep_time)  # can be replaced by some synchronization mechanism, don't mind now.
            sleep_time = min(sleep_time + .5, max_sleep_time)

    print('parallel part took: ', datetime.now() - start)

    # for file in pickle_file_dir.iterdir():
    #     bests = get_the_best([bests, *read_from_disk(file)])

    print(bests)
    print(*bests)
    for b in bests:
        indexes = b.data
        print(sort_code(codewords[indexes, :]).astype(int))

    write_to_disk(pickle_file_dir.with_suffix('.pkl'), bests)
    if not args.keep_dir:
        shutil.rmtree(pickle_file_dir, ignore_errors=True)

    # write the codes in human readable format
    write_readable_results(output_file, codewords, bests, (n, m, k))
