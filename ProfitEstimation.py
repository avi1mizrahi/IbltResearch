# parameters:
# - txs only in A
# - txs only in B
# - txs in A^B
# - parameters for IBLT (don't know what exactly, size, k, etc., do they should be a function of the set sizes)
# - k: number of iterations for success rate calculation

# no prios:
#  A send IBLT with all its transactions to B.
#  B constructs its IBLT the same way and subtruct from A's message.
#  What is the probability to decode the diffrence?

# with prios:
#  A and B know the exact partition.
#  Both do the same as in the prev simulation, but now for two sets:
#  1. IBLT for txs with P>t
#  2. txs with P<t - sends in plain

# The experiment:
# - the total size of the IBLTs in both scenarios should be equal.
# - generate the sets of txs (by the params) with probabilities P:tx->{0,1} for each.
# - run k times each of the scenarios, calculate for each the success rate.

# IBLT implementation: https://github.com/jesperborgstrup/Py-IBLT
import builtins
import itertools
import random
from multiprocessing import Pool
import math
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np

from iblt.pyblt import PYBLT


class Colors:
	active = True
	GREEN = '\033[92m'
	RED = '\033[91m'
	END = '\033[0m'

	@staticmethod
	def color(s, code):
		if not Colors.active:
			return s
		return code + s + Colors.END

	@staticmethod
	def red(s):
		return Colors.color(s, Colors.RED)

	@staticmethod
	def green(s):
		return Colors.color(s, Colors.GREEN)


VALUE_SIZE = 16
KEY_SIZE = 8
SIZE_DEFICIT_FRACTION = .96


def round(x):
	"""
	Make sure round returns int, as it may return numpy floats in some cases.
	"""
	return int(builtins.round(x))


def getTotalMemory(aOnlytxsNum, bOnlytxsNum, bothtxsNum):
	return PYBLT(entries=aOnlytxsNum + bOnlytxsNum, value_size=VALUE_SIZE).getIbltSize()


def partition(pred, iterable):
	"""Use a predicate to partition entries into false entries and true entries"""
	# partition(is_odd, range(10)) --> 0 2 4 6 8   and  1 3 5 7 9
	t1, t2 = itertools.tee(iterable)
	return itertools.filterfalse(pred, t1), filter(pred, t2)


def generateTxs(txsNum, nByts=6):
	# Generate random transactions, encode to utf in order to hash
	# Return list of tuples of key, value
	# TODO: Generate transactions to have VALUE_SIZE size
	return [(tx, str(tx)) for tx in random.sample(range(2 ** (nByts * 8)), txsNum)]  # sample without replacement


def noPriorsIteration(totalMemory, aOnlyTxs, bothTxs, bOnlyTxs, errorRate):
	aTxs = aOnlyTxs + bothTxs
	bTxs = bOnlyTxs + bothTxs

	# A constructs IBLT and send to B
	aIblt = PYBLT(value_size=VALUE_SIZE, ibltSize=totalMemory, num_hashes=3)
	for key, val in aTxs:
		aIblt.insert(key, val)

	# B subtracts from A IBLT
	# TODO: Use subtruction
	for key, val in bTxs:
		aIblt.erase(key, val)
	# B tries to decode the IBLT
	result, entries_ = aIblt.list_entries()
	return result

def withPriorsSendData(totalMemory, aTxs, threshold):
	""" 
	aTxs is all of A txs. Threshold is the index from which a think the txs are common for both a and b.
	Returns the data A sends to B
	"""
	aOnlyTxs = aTxs[:threshold]
	bothTxs = aTxs[threshold:]
	# A sends transactions that B doesnt know
	size_sent = len(aOnlyTxs) * (VALUE_SIZE + KEY_SIZE)
	iblt_size = totalMemory - size_sent
	if iblt_size <= 0:
		# In this case the total memory is too small
		return aOnlyTxs, None
	# A constructs IBLT for transactions that B knows
	# TODO: What should entries be?
	aIblt = PYBLT(value_size=VALUE_SIZE, ibltSize=iblt_size, num_hashes=3)
	for key, val in bothTxs:
		aIblt.insert(key, val)
		
	return aOnlyTxs, aIblt
	
def withPriorsGetData(aIblt, bTxs, threshold):
	"""
	Decrypts aIblt, return True if succeeded, else return False
	"""
	bOnlyTxs = bTxs[:threshold]
	bothTxs = bTxs[threshold:]
	# B subtracts from A IBLT
	for key, val in bothTxs:
		aIblt.erase(key, val)
	# B tries to decode the IBLT
	result, _ = aIblt.list_entries()
	# Return True if the IBLT was decoded. else return False
	return result

def withPriorsIteration(totalMemory, aOnlyTxs, bothTxs, bOnlyTxs, errorRate):
	# TODO: Should I move some txs from bothTxs?
	aTxs = aOnlyTxs + bothTxs
	aOnlyThreshold = round((1 - errorRate) * len(aOnlyTxs))
	_, aIblt = withPriorsSendData(totalMemory, aTxs, aOnlyThreshold)
	if not aIblt:
		return False
		
	bTxs = bOnlyTxs + bothTxs
	bOnlyThreshold = round((1 - errorRate) * len(bOnlyTxs))
	return withPriorsGetData(aIblt, bTxs, bOnlyThreshold)   

def run(iterationNum=150,
		aOnlytxsNum=100,
		bOnlytxsNum=0,
		bothtxsNum=200,
		errorRate=0,
		idealMemoryFrac=SIZE_DEFICIT_FRACTION,
		totalMemory=None):

	PYBLT.set_parameter_filename('iblt/param.export.csv')

	if not totalMemory:
		# Check the ideal memory, and return a bit less
		totalMemory = round(getTotalMemory(aOnlytxsNum, bOnlytxsNum, bothtxsNum) * idealMemoryFrac)
	successCounters = [0, 0]
	assert totalMemory >= PYBLT.getEntrySize(VALUE_SIZE), f'{totalMemory} < {PYBLT.getEntrySize(VALUE_SIZE)}'
	for i in range(iterationNum):
		# all -> [ aOnlytxsNum  | bothtxsNum | bOnlytxsNum    ]
		allTxs = generateTxs(bothtxsNum + aOnlytxsNum + bOnlytxsNum)
		aOnlyTxs = allTxs[:aOnlytxsNum]
		bothTxs = allTxs[aOnlytxsNum:aOnlytxsNum + bothtxsNum]
		bOnlyTxs = allTxs[aOnlytxsNum + bothtxsNum:]

		for i, f in enumerate([noPriorsIteration, withPriorsIteration]):
			result = f(totalMemory=totalMemory, aOnlyTxs=aOnlyTxs, bOnlyTxs=bOnlyTxs, bothTxs=bothTxs, errorRate=errorRate)
			if result:
				successCounters[i] += 1
	return tuple(c / iterationNum for c in successCounters)


# this is just a wrapper for convenience, can be merged into the "run" function.
def experiment(totalTxsNum, bothFraction, idealMemoryFrac, errorRate):
	# Determine how many txs are aOnly and bOnly
	bothtxsNum = round(totalTxsNum * bothFraction)
	aOnlytxsNum = math.ceil((totalTxsNum - bothtxsNum) / 2)
	bOnlytxsNum = math.floor((totalTxsNum - bothtxsNum) / 2)

	rates = run(
		aOnlytxsNum=aOnlytxsNum,
		bothtxsNum=bothtxsNum,
		bOnlytxsNum=bOnlytxsNum,
		idealMemoryFrac=idealMemoryFrac,
		errorRate=errorRate
	)

	badge = ''
	if rates[0] < rates[1]:
		badge = Colors.green('WIN')
	if rates[0] > rates[1]:
		badge = Colors.red('LOS')

	print(f'total={totalTxsNum:4}; both={bothFraction:0.2f}; memoryFrac={idealMemoryFrac:0.2f}; errorRate={errorRate:0.2f}; rates={rates[0]:0.2f}:{rates[1]:0.2f} [{badge:3}]')
	return {
		'totalTxsNum': totalTxsNum, 
		'bothFraction': bothFraction, 
		'aOnlytxsNum': aOnlytxsNum, 
		'bOnlytxsNum': bOnlytxsNum, 
		'idealMemoryFrac': idealMemoryFrac, 
		'errorRate': errorRate,
		'noPriorSuccessRate': rates[0],
		'withPriorSuccessRate': rates[1],
	}


if __name__ == "__main__":
	totalTxsNum = [10**5, 10**4, 10**3, 10**2]
	bothFraction = np.linspace(start=0.05, stop=0.95, num=19)
	idealMemoryFrac = np.linspace(start=0.85, stop=1.05, num=17)
	errorRate = np.linspace(start=0, stop=1, num=21)

	experiment_params = itertools.product(
			totalTxsNum, bothFraction, idealMemoryFrac, errorRate, 
	)

	# run each experiment in different process
	with Pool(maxtasksperchild=2) as pool:
		results = pool.starmap(experiment, experiment_params, chunksize=2)

	resTable = pd.DataFrame(results)
	resTable.to_csv('iblt_split_rates.csv', index=False)

	# draw success rate as function  of idealMemoryFrac
	plt.figure(1)
	plt.xlabel("idealMemoryFrac")
	plt.ylabel("success rate")
	plt.plot(resTable['idealMemoryFrac'], resTable['noPriorSuccessRate'], 'ro')
	plt.plot(resTable['idealMemoryFrac'], resTable['withPriorSuccessRate'], 'bx')
	plt.show()

	# draw success rate as function  of error rate
	plt.figure(2)
	plt.xlabel("error rate")
	plt.ylabel("success rate")
	plt.plot(resTable['errorRate'], resTable['noPriorSuccessRate'], 'ro')
	plt.plot(resTable['errorRate'], resTable['withPriorSuccessRate'], 'bx')
	plt.show()
