import struct

entry = struct.Struct('<Q32s3qQB')


def bytes2hex(bytes):
	return hex(int.from_bytes(bytes, 'little'))


reason = {
	0: 'EXPIRY',  # Expired from mempool
	1: 'SIZELIMIT',  # Removed in size limiting
	2: 'REORG',  # Removed for reorganization
	3: 'BLOCK',  # Removed for block
	4: 'CONFLICT',  # Removed for conflict with in-block transaction
	5: 'REPLACED',  # Removed for replacement
	255: 'INSERTED',
}


def transform(infilename, outfilename):
	with open(infilename, 'rb') as infile, open(outfilename, 'w') as outfile:
		print('i', 'hash', 'time', 'fee', 'fee_delta', 'size', 'reason', sep=',', file=outfile)

		buf = infile.read(entry.size)
		while len(buf) == entry.size:
			i, h, t, f, df, s, r = entry.unpack(buf)
			print(i, bytes2hex(h), t, f, df, s, reason[r], sep=',', file=outfile)
			buf = infile.read(entry.size)


if __name__ == '__main__':
	from sys import argv
	transform(*argv[1:1+2])
