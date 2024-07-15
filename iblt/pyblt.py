"""Python wrapper for Gavin Andresen's c++-based IBLT implementation."""

__author__ = "Brian Levine"

import itertools
from ctypes import *
from collections import defaultdict

import pathlib
import sys
assert sys.version_info.major > 2, 'This code requires python 3 or above.'
LIB = cdll.LoadLibrary(pathlib.Path(__file__).parent / 'libpyblt.so')


class PYBLT():
    """
    Uses  All the functions take the pointer to the allocated c++ data Structure
    as the first argument.  (c_ulong is the pointer.)
    """

    def __init__(self, value_size, entries=None, hedge=None, num_hashes=None, allocate=True,  ibltSize=None, m=None):
        """
        entries: number of items to be recovered from the IBLT (not necessarily the number inserted)
        value_size: all stored values must be the same size
        hedge: If None, then autoset by c++; a multiplier on the number of cells.
        num_hashes: If None, then autoset by c++; number of hash functions used by IBLT.
        allocate: used internally, if False, don't call c++ function to allocate memory
        ibltSize: If not None - determined the size of the iblt. can be used only when hedge and num_hashes are None.
        m: actual number of cells in the IBLT
        """

        self.value_size = value_size
        self.POINTER = 0

        if hedge is not None and num_hashes is not None:
            assert(entries is not None)
            call = LIB.pyblt_manual
            call.argtypes = [c_int, c_int, c_float, c_int]
            call.restype = c_ulong
            if allocate:
                self.POINTER = call(entries, value_size, hedge, num_hashes)
        elif hedge is None and num_hashes is None and ibltSize is None:
            call = LIB.pyblt_new
            call.argtypes = [c_int, c_int]
            call.restype = c_ulong
            if allocate:
                self.POINTER = call(entries, value_size)
        elif hedge is None and num_hashes is not None and ibltSize is not None:
            assert(entries is None)
            call = LIB.pyblt_fixed_size
            call.argtypes = [c_size_t, c_size_t, c_size_t]
            call.restype = c_ulong
            if allocate:
                self.POINTER = call(value_size, ibltSize, num_hashes)
        elif m is not None:
            call = LIB.pyblt_normal
            call.argtypes = [c_size_t, c_size_t, c_size_t]
            call.restype = c_ulong
            if allocate:
                self.POINTER = call(value_size, m, num_hashes)

    def __del__(self):
        """ Deallocate memory"""
        call = LIB.pyblt_delete
        call.argtypes = [c_ulong]
        call(self.POINTER)

    @staticmethod
    def set_parameter_filename(filename):
        """ Set the csv file to use that optimizes parameters """
        call = LIB.pyblt_set_parameter_file
        call.argtypes = [c_char_p]
        call(bytes(filename, "ascii"))

    def getNumHashes(self):
        """ Get number of hashes in the IBLT"""
        call = LIB.getNumHashes
        call.argtypes = [c_ulong]
        call.restype = c_int
        result = call(self.POINTER)
        return result

    def getIbltSize(self):
        """ Get size of the IBLT"""
        call = LIB.getIbltSize
        call.argtypes = [c_ulong]
        call.restype = c_uint
        result = call(self.POINTER)
        return result

    def clear(self):
        """ Clear the table """
        call = LIB.clear
        call.argtypes = [c_ulong]
        call(self.POINTER)

    @staticmethod
    def getEntrySize(value_size):
        """ Get size of the IBLT"""
        call = LIB.getEntrySize
        call.argtypes = [c_uint]
        call.restype = c_uint
        result = call(value_size)
        return result

    def dump_table(self):
        """ Dump the internal representatino of the IBLT"""
        call = LIB.pyblt_dump_table
        call.argtypes = [c_ulong]
        call.restype = c_char_p
        result = call(self.POINTER)
        result = [x.decode() for x in result.split(b'\n')]
        return([x for x in result if x != ''])

    def insert(self, key_int, value=''):
        """ Insert a new (key,value) pair.
        key_int: key must be an integer
        value: anything you want, but it will be converted to a string before storage
        """
        value = bytes(str("%"+str(self.value_size)+"s") % value, "ascii")

        assert type(value) == type(b''), "value is not a bytearray:  %s != %s" % (type(value), type(b''))
        assert len(value) == self.value_size, "value size is %d != %d" % (len(value), self.value_size)

        call = LIB.pyblt_insert
        call.argtypes = [c_ulong, c_ulong, c_char_p]
        call(self.POINTER, key_int, bytes(value.hex(), 'ascii'))

    def erase(self, key_int, value=''):
        """ Erase a new (key,value) pair.
        key_int: key must be an integer
        value: anything you want, but it will be converted to a string before storage
        """
        value = bytes(str("%"+str(self.value_size)+"s") % value, 'ascii')
        assert type(value) == type(b''), "value is not a bytearray:  %s != %s" % (type(value), type(b''))
        assert len(value) == self.value_size, "value size is %d != %d" % (len(value), self.value_size)

        call = LIB.pyblt_erase
        call.argtypes = [c_ulong, c_ulong, c_char_p]
        call(self.POINTER, key_int, bytes(value.hex(), 'ascii'))

    class RESULT(Structure):
        """ Return results from the IBLT that contain keys and values """
        # This must be in the same order as the C++ struct
        _fields_ = [
            ('decoded', c_bool),
            ("pos_len", c_uint),
            ("neg_len", c_uint),
            ("pos_keys", POINTER(c_ulonglong)),
            ("neg_keys", POINTER(c_ulonglong)),
            ("pos_str", c_char_p),
            ("neg_str", c_char_p)]

    class KEYS(Structure):
        """ Return results from the IBLT that contain keys only, no values """
        # This must be in the same order as the C++ struct
        _fields_ = [
            ("pos_len", c_uint),
            ("neg_len", c_uint),
            ("pos_keys", POINTER(c_ulonglong)),
            ("neg_keys", POINTER(c_ulonglong))]

    @staticmethod
    def decode_list_results(l, keys, values, entries, direction):
        if not values:
            values = itertools.repeat('')

        for key, value in itertools.islice(zip(keys, values), l):
            entries[key] = (bytes.fromhex(value).decode(), direction)

    def list_entries(self):
        """ Decode the IBLT and list all entries (keys and values).
        The IBLT is left in tact.
        This is fragile in that a partially decodeable IBLT just returns None
        """
        call = LIB.pyblt_list_entries
        call.argtypes = [c_ulong]
        call.restype = self.RESULT
        result = call(self.POINTER)

        entries = defaultdict(dict)
        self.decode_list_results(result.pos_len, result.pos_keys, result.pos_str.decode().split(), entries, 1)
        self.decode_list_results(result.neg_len, result.neg_keys, result.neg_str.decode().split(), entries, -1)

        return result.decoded, entries

    def peel(self):
        """ Peel the IBLT in place. Meaning, any found entries are removed from the tableself.
        This function returns only keys, not values.
        """
        call = LIB.pyblt_peel_entries
        call.argtypes = [c_ulong]
        call.restype = self.KEYS
        result = call(self.POINTER)
        # print("pos len",result.pos_len)
        # print("neg len",result.neg_len)
        entries = list()
        for x in range(result.pos_len):
            entries += [result.pos_keys[x]]
        for x in range(result.neg_len):
            entries += [result.neg_keys[x]]
        return entries

    def subtract(self, other):
        """ call an "other" IBLT from this one, as per Eppstein. Not destructive. """
        assert type(other) == type(PYBLT(1, 1, 1, 2))        
        call = LIB.pyblt_subtract
        call.argtypes = [c_ulong, c_ulong]
        call.restype = c_ulong
        #the call to subtract allocates new memory, we need to receive it.
        res = PYBLT(entries=None, value_size=self.value_size, allocate=False)  # Nones are ignored when last argument is false
        res.POINTER = call(self.POINTER, other.POINTER)
        return res

    def get_serialized_size(self):
        """ Minimal size of the IBLT if serialized, in count of cells (rows). Doesn't include parameters. """
        call = LIB.pyblt_capacity
        call.argtypes = [c_ulong]
        call.restype = c_int
        res = call(self.POINTER)
        return res
