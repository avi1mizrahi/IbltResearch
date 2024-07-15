from itertools import count, takewhile, islice

import attr

from itertools_recipes import first_true


class Generator:
    def __init__(self):
        self.primes = [1]

    def __iter__(self):
        for i in count(1):
            yield self.primes[i] if i < len(self.primes) else next(self)

    def __next__(self):
        candidate = self.primes[-1] + 1
        while True:
            for prime in takewhile(lambda p: candidate >= p ** 2,
                                   islice(self.primes, 1, None)):
                if candidate % prime == 0:
                    break
            else:
                self.primes.append(candidate)
                break

            candidate += 1

        return self.primes[-1]

    def __getitem__(self, i):
        """ returns the i'th prime, 0->2, 1->3, 2->5, ..."""
        i += 1  # convention
        while i >= len(self.primes):
            next(self)
        return self.primes[i]


@attr.s
class Power:
    pg = attr.ib(factory=Generator)

    def decompose(self, n):
        for p in self.pg:
            if n % p != 0:
                continue
            a = count()
            while n % p == 0:
                next(a)
                n //= p
            if n == 1:
                return p, next(a)
            else:
                return None

    def is_prime_power(self, n):
        return self.decompose(n) is not None

    def closest(self, n):
        return first_true(count(start=n), pred=self.is_prime_power)

    def __iter__(self):
        for q in count(start=2):
            if self.is_prime_power(q):
                yield q
