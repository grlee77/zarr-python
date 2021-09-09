import atexit
import os
import sys
import pickle
import shutil
import unittest
from itertools import zip_longest
from tempfile import mkdtemp, mktemp

import numpy as np
import pytest
from numcodecs import (BZ2, JSON, LZ4, Blosc, Categorize, Delta,
                       FixedScaleOffset, GZip, MsgPack, Pickle, VLenArray,
                       VLenBytes, VLenUTF8, Zlib)
from numcodecs.compat import ensure_bytes, ensure_ndarray
from numcodecs.tests.common import greetings
from numpy.testing import assert_array_almost_equal, assert_array_equal
from pkg_resources import parse_version

from zarr.core import Array
from zarr.errors import ArrayNotFoundError, ContainsGroupError
from zarr.indexing import BoundsCheckError
from zarr.meta import json_loads, ZARR_V3_ALLOW_COMPLEX, ZARR_V3_ALLOW_DATETIME
from zarr.n5 import N5Store, n5_keywords
from zarr.storage import (
    # ABSStoreV3,
    DBMStoreV3,
    DirectoryStoreV3,
    FSStoreV3,
    KVStoreV3,
    LMDBStoreV3,
    LRUStoreCacheV3,
    NestedDirectoryStoreV3,
    SQLiteStoreV3,
    StoreV3,
    atexit_rmglob,
    atexit_rmtree,
    init_array,
    init_group,
)
from zarr.tests.test_core import TestArrayWithPath
from zarr.tests.util import abs_container, skip_test_env_var, have_fsspec
from zarr.util import buffer_size


# Start with TestArrayWithPathV3 not TestArrayV3 since path must be supplied

class TestArrayWithPathV3(TestArrayWithPath):

    _version = 3

    @staticmethod
    def create_array(array_path='arr1', read_only=False, **kwargs):
        store = KVStoreV3(dict())
        kwargs.setdefault('compressor', Zlib(level=1))
        cache_metadata = kwargs.pop('cache_metadata', True)
        cache_attrs = kwargs.pop('cache_attrs', True)
        init_array(store, path=array_path, **kwargs)
        return Array(store, path=array_path, read_only=read_only,
                     cache_metadata=cache_metadata, cache_attrs=cache_attrs)

    def test_array_init(self):

        # should not be able to initialize without a path in V3
        store = KVStoreV3(dict())
        with pytest.raises(ValueError):
            init_array(store, shape=100, chunks=10, dtype="<f8")

        # initialize at path
        store = KVStoreV3(dict())
        path = 'foo/bar'
        init_array(store, shape=100, chunks=10, path=path, dtype='<f8')
        a = Array(store, path=path)
        assert isinstance(a, Array)
        assert (100,) == a.shape
        assert (10,) == a.chunks
        assert path == a.path  # TODO: should this include meta/root?
        assert '/' + path == a.name  # TODO: should this include meta/root?
        assert 'bar' == a.basename
        assert store is a.store
        assert "fde0ede82971732ad9fda9c042a1ce3cc955edc6" == a.hexdigest()

        # store not initialized
        store = KVStoreV3(dict())
        with pytest.raises(ValueError):
            Array(store)

        # group is in the way
        store = KVStoreV3(dict())
        path = 'baz'
        init_group(store, path=path)
        # can't open with an uninitialized array
        with pytest.raises(ArrayNotFoundError):
            Array(store, path=path)
        # can't open at same path as an existing group
        with pytest.raises(ContainsGroupError):
            init_array(store, shape=100, chunks=10, path=path, dtype='<f8')
        group_key = 'meta/root/' + path + '.group.json'
        assert group_key in store
        del store[group_key]
        init_array(store, shape=100, chunks=10, path=path, dtype='<f8')
        Array(store, path=path)
        assert group_key not in store
        assert ('meta/root/' + path + '.array.json') in store

    def test_nbytes_stored(self):

        # dict as store
        z = self.create_array(shape=1000, chunks=100)
        expect_nbytes_stored = sum(buffer_size(v) for k, v in z.store.items() if k != 'zarr.json')
        assert expect_nbytes_stored == z.nbytes_stored
        z[:] = 42
        expect_nbytes_stored = sum(buffer_size(v) for k, v in z.store.items() if k != 'zarr.json')
        assert expect_nbytes_stored == z.nbytes_stored
        assert z.nchunks_initialized == 10  # TODO: added temporarily for testing, can remove

        # mess with store
        try:
            z.store['data/root/' + z._key_prefix + 'foo'] = list(range(10))
            assert -1 == z.nbytes_stored
        except TypeError:
            pass

        z.store.close()

    def test_attributes(self):
        a = self.create_array(shape=10, chunks=10, dtype='i8')
        a.attrs['foo'] = 'bar'
        assert a.attrs.key in a.store
        assert 'foo' in a.attrs and a.attrs['foo'] == 'bar'
        # in v3, attributes are in a sub-dictionary
        attrs = json_loads(a.store[a.attrs.key])['attributes']
        assert 'foo' in attrs and attrs['foo'] == 'bar'

        a.attrs['bar'] = 'foo'
        assert a.attrs.key in a.store
        # in v3, attributes are in a sub-dictionary
        attrs = json_loads(a.store[a.attrs.key])['attributes']
        assert 'foo' in attrs and attrs['foo'] == 'bar'
        assert 'bar' in attrs and attrs['bar'] == 'foo'
        a.store.close()

    def test_dtypes(self):

        # integers
        for dtype in 'u1', 'u2', 'u4', 'u8', 'i1', 'i2', 'i4', 'i8':
            z = self.create_array(shape=10, chunks=3, dtype=dtype)
            assert z.dtype == np.dtype(dtype)
            a = np.arange(z.shape[0], dtype=dtype)
            z[:] = a
            assert_array_equal(a, z[:])
            z.store.close()

        # floats
        for dtype in 'f2', 'f4', 'f8':
            z = self.create_array(shape=10, chunks=3, dtype=dtype)
            assert z.dtype == np.dtype(dtype)
            a = np.linspace(0, 1, z.shape[0], dtype=dtype)
            z[:] = a
            assert_array_almost_equal(a, z[:])
            z.store.close()

        # TODO: should v3 spec be extended to include these complex and
        #       datetime dtypes?

        # complex
        for dtype in 'c8', 'c16':
            if not ZARR_V3_ALLOW_COMPLEX:
                with pytest.raises(ValueError):
                    self.create_array(shape=10, chunks=3, dtype=dtype)
                return
            z = self.create_array(shape=10, chunks=3, dtype=dtype)
            assert z.dtype == np.dtype(dtype)
            a = np.linspace(0, 1, z.shape[0], dtype=dtype)
            a -= 1j * a
            z[:] = a
            assert_array_almost_equal(a, z[:])
            z.store.close()

        # datetime, timedelta
        for base_type in 'Mm':
            for resolution in 'D', 'us', 'ns':
                dtype = '{}8[{}]'.format(base_type, resolution)
                if not ZARR_V3_ALLOW_DATETIME:
                    with pytest.raises(ValueError):
                        self.create_array(shape=100, dtype=dtype, fill_value=0)
                    return
                z = self.create_array(shape=100, dtype=dtype, fill_value=0)
                assert z.dtype == np.dtype(dtype)
                a = np.random.randint(np.iinfo('i8').min, np.iinfo('i8').max,
                                      size=z.shape[0],
                                      dtype='i8').view(dtype)
                z[:] = a
                assert_array_equal(a, z[:])
                z.store.close()

        # check that datetime generic units are not allowed
        with pytest.raises(ValueError):
            self.create_array(shape=100, dtype='M8')
        with pytest.raises(ValueError):
            self.create_array(shape=100, dtype='m8')

    def expected(self):
        return [
            "f8b6f297c56e42cb086f3b020327402801f0de0f",
            "36c94c6d0933f3e0ced633047bffb7bd8c4c99ec",
            "02ed16cf33b402cd9b84c2a33db7ffcd2e678a17",
            "d9f57d9a66b7bd249f5dd3ccab56475b53e203e7",
            "de74d165181c090743246748f154394186db378a",
        ]

    def test_hexdigest(self):
        found = []

        # Check basic 1-D array
        z = self.create_array(shape=(1050,), chunks=100, dtype='<i4')
        found.append(z.hexdigest())
        z.store.close()

        # Check basic 1-D array with different type
        z = self.create_array(shape=(1050,), chunks=100, dtype='<f4')
        found.append(z.hexdigest())
        z.store.close()

        # Check basic 2-D array
        z = self.create_array(shape=(20, 35,), chunks=10, dtype='<i4')
        found.append(z.hexdigest())
        z.store.close()

        # Check basic 1-D array with some data
        z = self.create_array(shape=(1050,), chunks=100, dtype='<i4')
        z[200:400] = np.arange(200, 400, dtype='i4')
        found.append(z.hexdigest())
        z.store.close()

        # # Check basic 1-D array with attributes
        z = self.create_array(shape=(1050,), chunks=100, dtype='<i4')
        z.attrs['foo'] = 'bar'
        found.append(z.hexdigest())
        z.store.close()
        print(f"found = {found}")
        print(f"self.expected() = {self.expected()}")

        assert self.expected() == found

    def test_nchunks_initialized(self):
        # copied from TestArray so the empty version from TestArrayWithPath is
        # not used

        z = self.create_array(shape=100, chunks=10)
        assert 0 == z.nchunks_initialized
        # manually put something into the store to confuse matters
        z.store['foo'] = b'bar'
        assert 0 == z.nchunks_initialized
        z[:] = 42
        assert 10 == z.nchunks_initialized

        z.store.close()


class TestArrayWithChunkStoreV3(TestArrayWithPathV3):

    @staticmethod
    def create_array(array_path='arr1', read_only=False, **kwargs):
        store = KVStoreV3(dict())
        # separate chunk store
        chunk_store = KVStoreV3(dict())
        cache_metadata = kwargs.pop('cache_metadata', True)
        cache_attrs = kwargs.pop('cache_attrs', True)
        init_array(store, path=array_path, chunk_store=chunk_store, **kwargs)
        return Array(store, path=array_path, read_only=read_only,
                     chunk_store=chunk_store, cache_metadata=cache_metadata,
                     cache_attrs=cache_attrs)

    def test_hexdigest(self):
        # Check basic 1-D array
        z = self.create_array(shape=(1050,), chunks=100, dtype='<i4')
        assert 'd8c6501914fd9e999a437c304cfe8c8f4a47f3a1' == z.hexdigest()

        # Check basic 1-D array with different type
        z = self.create_array(shape=(1050,), chunks=100, dtype='<f4')
        assert 'b52ac4f6f1982611d9d4b9e32f027603f7076e15' == z.hexdigest()

        # Check basic 2-D array
        z = self.create_array(shape=(20, 35,), chunks=10, dtype='<i4')
        assert '936b531463d9a7575c5766cb5fe6e9a85598fc4e' == z.hexdigest()

        # Check basic 1-D array with some data
        z = self.create_array(shape=(1050,), chunks=100, dtype='<i4')
        z[200:400] = np.arange(200, 400, dtype='i4')
        assert 'b9fc48400b71f30de731dcd838b49d1bdbe3b21b' == z.hexdigest()

        # Check basic 1-D array with attributes
        z = self.create_array(shape=(1050,), chunks=100, dtype='<i4')
        z.attrs['foo'] = 'bar'
        assert 'dccb51bd549886fb676c255c99f78a437a5b6cd8' == z.hexdigest()

    def test_nbytes_stored(self):

        z = self.create_array(shape=1000, chunks=100)
        expect_nbytes_stored = sum(buffer_size(v) for k, v in z.store.items() if k != 'zarr.json')
        expect_nbytes_stored += sum(buffer_size(v)
                                    for k, v in z.chunk_store.items() if k != 'zarr.json')
        assert expect_nbytes_stored == z.nbytes_stored
        z[:] = 42
        expect_nbytes_stored = sum(buffer_size(v) for k, v in z.store.items() if k != 'zarr.json')
        expect_nbytes_stored += sum(buffer_size(v)
                                    for k, v in z.chunk_store.items() if k != 'zarr.json')
        assert expect_nbytes_stored == z.nbytes_stored

        # mess with store
        z.chunk_store['data/root/' + z._key_prefix + 'foo'] = list(range(10))
        assert -1 == z.nbytes_stored


class TestArrayWithDirectoryStoreV3(TestArrayWithPathV3):

    @staticmethod
    def create_array(array_path='arr1', read_only=False, **kwargs):
        path = mkdtemp()
        atexit.register(shutil.rmtree, path)
        store = DirectoryStoreV3(path)
        cache_metadata = kwargs.pop('cache_metadata', True)
        cache_attrs = kwargs.pop('cache_attrs', True)
        kwargs.setdefault('compressor', Zlib(1))
        init_array(store, path=array_path, **kwargs)
        return Array(store, path=array_path, read_only=read_only,
                     cache_metadata=cache_metadata, cache_attrs=cache_attrs)

    def test_nbytes_stored(self):
        # dict as store
        z = self.create_array(shape=1000, chunks=100)
        expect_nbytes_stored = sum(buffer_size(v) for k, v in z.store.items() if k != 'zarr.json')
        assert expect_nbytes_stored == z.nbytes_stored
        z[:] = 42
        expect_nbytes_stored = sum(buffer_size(v) for k, v in z.store.items() if k != 'zarr.json')
        assert expect_nbytes_stored == z.nbytes_stored


# TODO: TestArrayWithABSStoreV3
# @skip_test_env_var("ZARR_TEST_ABS")
# class TestArrayWithABSStoreV3(TestArrayWithPathV3):


class TestArrayWithNestedDirectoryStoreV3(TestArrayWithDirectoryStoreV3):

    @staticmethod
    def create_array(array_path='arr1', read_only=False, **kwargs):
        path = mkdtemp()
        atexit.register(shutil.rmtree, path)
        store = NestedDirectoryStoreV3(path)
        cache_metadata = kwargs.pop('cache_metadata', True)
        cache_attrs = kwargs.pop('cache_attrs', True)
        kwargs.setdefault('compressor', Zlib(1))
        init_array(store, path=array_path, **kwargs)
        return Array(store, path=array_path, read_only=read_only, cache_metadata=cache_metadata,
                     cache_attrs=cache_attrs)

    def expected(self):
        return [
            "f8b6f297c56e42cb086f3b020327402801f0de0f",
            "36c94c6d0933f3e0ced633047bffb7bd8c4c99ec",
            "02ed16cf33b402cd9b84c2a33db7ffcd2e678a17",
            "d9f57d9a66b7bd249f5dd3ccab56475b53e203e7",
            "de74d165181c090743246748f154394186db378a",
        ]


# TODO: TestArrayWithN5StoreV3
# class TestArrayWithN5StoreV3(TestArrayWithDirectoryStoreV3):


class TestArrayWithDBMStoreV3(TestArrayWithPathV3):

    @staticmethod
    def create_array(array_path='arr1', read_only=False, **kwargs):
        path = mktemp(suffix='.anydbm')
        atexit.register(atexit_rmglob, path + '*')
        store = DBMStoreV3(path, flag='n')
        cache_metadata = kwargs.pop('cache_metadata', True)
        cache_attrs = kwargs.pop('cache_attrs', True)
        kwargs.setdefault('compressor', Zlib(1))
        init_array(store, path=array_path, **kwargs)
        return Array(store, path=array_path, read_only=read_only, cache_attrs=cache_attrs,
                     cache_metadata=cache_metadata)

    def test_nbytes_stored(self):
        pass  # not implemented


class TestArrayWithDBMStoreV3BerkeleyDB(TestArrayWithPathV3):

    @staticmethod
    def create_array(array_path='arr1', read_only=False, **kwargs):
        bsddb3 = pytest.importorskip("bsddb3")
        path = mktemp(suffix='.dbm')
        atexit.register(os.remove, path)
        store = DBMStoreV3(path, flag='n', open=bsddb3.btopen)
        cache_metadata = kwargs.pop('cache_metadata', True)
        cache_attrs = kwargs.pop('cache_attrs', True)
        kwargs.setdefault('compressor', Zlib(1))
        init_array(store, path=array_path, **kwargs)
        return Array(store, path=array_path, read_only=read_only, cache_metadata=cache_metadata,
                     cache_attrs=cache_attrs)

    def test_nbytes_stored(self):
        pass  # not implemented


class TestArrayWithLMDBStoreV3(TestArrayWithPathV3):

    @staticmethod
    def create_array(array_path='arr1', read_only=False, **kwargs):
        pytest.importorskip("lmdb")
        path = mktemp(suffix='.lmdb')
        atexit.register(atexit_rmtree, path)
        store = LMDBStoreV3(path, buffers=True)
        cache_metadata = kwargs.pop('cache_metadata', True)
        cache_attrs = kwargs.pop('cache_attrs', True)
        kwargs.setdefault('compressor', Zlib(1))
        init_array(store, path=array_path, **kwargs)
        return Array(store, path=array_path, read_only=read_only, cache_metadata=cache_metadata,
                     cache_attrs=cache_attrs)

    def test_store_has_bytes_values(self):
        pass  # returns values as memoryviews/buffers instead of bytes

    def test_nbytes_stored(self):
        pass  # not implemented


class TestArrayWithLMDBStoreV3NoBuffers(TestArrayWithPathV3):

    @staticmethod
    def create_array(array_path='arr1', read_only=False, **kwargs):
        pytest.importorskip("lmdb")
        path = mktemp(suffix='.lmdb')
        atexit.register(atexit_rmtree, path)
        store = LMDBStoreV3(path, buffers=False)
        cache_metadata = kwargs.pop('cache_metadata', True)
        cache_attrs = kwargs.pop('cache_attrs', True)
        kwargs.setdefault('compressor', Zlib(1))
        init_array(store, path=array_path, **kwargs)
        return Array(store, path=array_path, read_only=read_only, cache_metadata=cache_metadata,
                     cache_attrs=cache_attrs)

    def test_nbytes_stored(self):
        pass  # not implemented


class TestArrayWithSQLiteStoreV3(TestArrayWithPathV3):

    @staticmethod
    def create_array(array_path='arr1', read_only=False, **kwargs):
        pytest.importorskip("sqlite3")
        path = mktemp(suffix='.db')
        atexit.register(atexit_rmtree, path)
        store = SQLiteStoreV3(path)
        cache_metadata = kwargs.pop('cache_metadata', True)
        cache_attrs = kwargs.pop('cache_attrs', True)
        kwargs.setdefault('compressor', Zlib(1))
        init_array(store, path=array_path, **kwargs)
        return Array(store, path=array_path, read_only=read_only, cache_metadata=cache_metadata,
                     cache_attrs=cache_attrs)

    def test_nbytes_stored(self):
        pass  # not implemented


# skipped adding V3 equivalents for compressors (no change in v3):
#    TestArrayWithNoCompressor
#    TestArrayWithBZ2Compressor
#    TestArrayWithBloscCompressor
#    TestArrayWithLZMACompressor

# skipped test with filters  (v3 protocol removed filters)
#    TestArrayWithFilters


# custom store, does not support getsize()
# Note: this custom mapping doesn't actually have all methods in the
#       v3 spec (e.g. erase), but they aren't needed here.
class CustomMappingV3(StoreV3):

    def __init__(self):
        self.inner = KVStoreV3(dict())

    def __iter__(self):
        return iter(self.keys())

    def __len__(self):
        return len(self.inner)

    def keys(self):
        return self.inner.keys()

    def values(self):
        return self.inner.values()

    def get(self, item, default=None):
        try:
            return self.inner[item]
        except KeyError:
            return default

    def __getitem__(self, item):
        return self.inner[item]

    def __setitem__(self, item, value):
        self.inner[item] = ensure_bytes(value)

    def __delitem__(self, key):
        del self.inner[key]

    def __contains__(self, item):
        return item in self.inner


class TestArrayWithCustomMappingV3(TestArrayWithPathV3):

    @staticmethod
    def create_array(array_path='arr1', read_only=False, **kwargs):
        store = CustomMappingV3()
        kwargs.setdefault('compressor', Zlib(1))
        cache_metadata = kwargs.pop('cache_metadata', True)
        cache_attrs = kwargs.pop('cache_attrs', True)
        init_array(store, path=array_path, **kwargs)
        return Array(store, path=array_path, read_only=read_only, cache_metadata=cache_metadata,
                     cache_attrs=cache_attrs)

    def test_nbytes_stored(self):
        z = self.create_array(shape=1000, chunks=100)
        expect_nbytes_stored = sum(buffer_size(v) for k, v in z.store.items() if k != 'zarr.json')
        assert expect_nbytes_stored == z.nbytes_stored
        z[:] = 42
        expect_nbytes_stored = sum(buffer_size(v) for k, v in z.store.items() if k != 'zarr.json')
        assert expect_nbytes_stored == z.nbytes_stored


class TestArrayNoCacheV3(TestArrayWithPathV3):

    @staticmethod
    def create_array(array_path='arr1', read_only=False, **kwargs):
        store = KVStoreV3(dict())
        kwargs.setdefault('compressor', Zlib(level=1))
        cache_metadata = kwargs.pop('cache_metadata', True)
        cache_attrs = kwargs.pop('cache_attrs', True)
        init_array(store, path=array_path, **kwargs)
        return Array(store, path=array_path, read_only=read_only, cache_metadata=cache_metadata,
                     cache_attrs=cache_attrs)

    def test_cache_metadata(self):
        a1 = self.create_array(shape=100, chunks=10, dtype='i1', cache_metadata=False)
        a2 = Array(a1.store, path=a1.path, cache_metadata=True)
        assert a1.shape == a2.shape
        assert a1.size == a2.size
        assert a1.nbytes == a2.nbytes
        assert a1.nchunks == a2.nchunks

        # a1 is not caching so *will* see updates made via other objects
        a2.resize(200)
        assert (200,) == a2.shape
        assert 200 == a2.size
        assert 200 == a2.nbytes
        assert 20 == a2.nchunks
        assert a1.shape == a2.shape
        assert a1.size == a2.size
        assert a1.nbytes == a2.nbytes
        assert a1.nchunks == a2.nchunks

        a2.append(np.zeros(100))
        assert (300,) == a2.shape
        assert 300 == a2.size
        assert 300 == a2.nbytes
        assert 30 == a2.nchunks
        assert a1.shape == a2.shape
        assert a1.size == a2.size
        assert a1.nbytes == a2.nbytes
        assert a1.nchunks == a2.nchunks

        # a2 is caching so *will not* see updates made via other objects
        a1.resize(400)
        assert (400,) == a1.shape
        assert 400 == a1.size
        assert 400 == a1.nbytes
        assert 40 == a1.nchunks
        assert (300,) == a2.shape
        assert 300 == a2.size
        assert 300 == a2.nbytes
        assert 30 == a2.nchunks

    # differs from v2 case only in that path='arr1' is passed to Array
    def test_cache_attrs(self):
        a1 = self.create_array(shape=100, chunks=10, dtype='i1', cache_attrs=False)
        a2 = Array(a1.store, path='arr1', cache_attrs=True)
        assert a1.attrs.asdict() == a2.attrs.asdict()

        # a1 is not caching so *will* see updates made via other objects
        a2.attrs['foo'] = 'xxx'
        a2.attrs['bar'] = 42
        assert a1.attrs.asdict() == a2.attrs.asdict()

        # a2 is caching so *will not* see updates made via other objects
        a1.attrs['foo'] = 'yyy'
        assert 'yyy' == a1.attrs['foo']
        assert 'xxx' == a2.attrs['foo']

    def test_object_arrays_danger(self):
        # skip this one as it only works if metadata are cached
        pass


class TestArrayWithStoreCacheV3(TestArrayWithPathV3):

    @staticmethod
    def create_array(array_path='arr1', read_only=False, **kwargs):
        store = LRUStoreCacheV3(dict(), max_size=None)
        kwargs.setdefault('compressor', Zlib(level=1))
        cache_metadata = kwargs.pop('cache_metadata', True)
        cache_attrs = kwargs.pop('cache_attrs', True)
        init_array(store, path=array_path, **kwargs)
        return Array(store, path=array_path, read_only=read_only, cache_metadata=cache_metadata,
                     cache_attrs=cache_attrs)

    def test_store_has_bytes_values(self):
        # skip as the cache has no control over how the store provides values
        pass


@pytest.mark.skipif(have_fsspec is False, reason="needs fsspec")
class TestArrayWithFSStoreV3(TestArrayWithPathV3):
    @staticmethod
    def create_array(array_path='arr1', read_only=False, **kwargs):
        path = mkdtemp()
        atexit.register(shutil.rmtree, path)
        key_separator = kwargs.pop('key_separator', ".")
        store = FSStoreV3(path, key_separator=key_separator, auto_mkdir=True)
        cache_metadata = kwargs.pop('cache_metadata', True)
        cache_attrs = kwargs.pop('cache_attrs', True)
        kwargs.setdefault('compressor', Blosc())
        init_array(store, path=array_path, **kwargs)
        return Array(store, path=array_path, read_only=read_only, cache_metadata=cache_metadata,
                     cache_attrs=cache_attrs)

    def expected(self):
        return [
           "d8c6501914fd9e999a437c304cfe8c8f4a47f3a1",
           "b52ac4f6f1982611d9d4b9e32f027603f7076e15",
           "936b531463d9a7575c5766cb5fe6e9a85598fc4e",
           "b9fc48400b71f30de731dcd838b49d1bdbe3b21b",
           "dccb51bd549886fb676c255c99f78a437a5b6cd8",
        ]

    def test_hexdigest(self):
        found = []

        # Check basic 1-D array
        z = self.create_array(shape=(1050,), chunks=100, dtype='<i4')
        found.append(z.hexdigest())

        # Check basic 1-D array with different type
        z = self.create_array(shape=(1050,), chunks=100, dtype='<f4')
        found.append(z.hexdigest())

        # Check basic 2-D array
        z = self.create_array(shape=(20, 35,), chunks=10, dtype='<i4')
        found.append(z.hexdigest())

        # Check basic 1-D array with some data
        z = self.create_array(shape=(1050,), chunks=100, dtype='<i4')
        z[200:400] = np.arange(200, 400, dtype='i4')
        found.append(z.hexdigest())

        # Check basic 1-D array with attributes
        z = self.create_array(shape=(1050,), chunks=100, dtype='<i4')
        z.attrs['foo'] = 'bar'
        found.append(z.hexdigest())

        assert self.expected() == found


@pytest.mark.skipif(have_fsspec is False, reason="needs fsspec")
class TestArrayWithFSStoreV3PartialRead(TestArrayWithPathV3):
    @staticmethod
    def create_array(array_path='arr1', read_only=False, **kwargs):
        path = mkdtemp()
        atexit.register(shutil.rmtree, path)
        store = FSStoreV3(path)
        cache_metadata = kwargs.pop("cache_metadata", True)
        cache_attrs = kwargs.pop("cache_attrs", True)
        kwargs.setdefault("compressor", Blosc())
        init_array(store, path=array_path, **kwargs)
        return Array(
            store,
            path=array_path,
            read_only=read_only,
            cache_metadata=cache_metadata,
            cache_attrs=cache_attrs,
            partial_decompress=True,
        )

    def test_hexdigest(self):
        # Check basic 1-D array
        z = self.create_array(shape=(1050,), chunks=100, dtype="<i4")
        assert "d8c6501914fd9e999a437c304cfe8c8f4a47f3a1" == z.hexdigest()

        # Check basic 1-D array with different type
        z = self.create_array(shape=(1050,), chunks=100, dtype="<f4")
        assert "b52ac4f6f1982611d9d4b9e32f027603f7076e15" == z.hexdigest()

        # Check basic 2-D array
        z = self.create_array(
            shape=(
                20,
                35,
            ),
            chunks=10,
            dtype="<i4",
        )
        assert "936b531463d9a7575c5766cb5fe6e9a85598fc4e" == z.hexdigest()

        # Check basic 1-D array with some data
        z = self.create_array(shape=(1050,), chunks=100, dtype="<i4")
        z[200:400] = np.arange(200, 400, dtype="i4")
        assert "b9fc48400b71f30de731dcd838b49d1bdbe3b21b" == z.hexdigest()

        # Check basic 1-D array with attributes
        z = self.create_array(shape=(1050,), chunks=100, dtype="<i4")
        z.attrs["foo"] = "bar"
        assert "dccb51bd549886fb676c255c99f78a437a5b6cd8" == z.hexdigest()

    def test_non_cont(self):
        z = self.create_array(shape=(500, 500, 500), chunks=(50, 50, 50), dtype="<i4")
        z[:, :, :] = 1
        # actually go through the partial read by accessing a single item
        assert z[0, :, 0].any()

    def test_read_nitems_less_than_blocksize_from_multiple_chunks(self):
        '''Tests to make sure decompression doesn't fail when `nitems` is
        less than a compressed block size, but covers multiple blocks
        '''
        z = self.create_array(shape=1000000, chunks=100_000)
        z[40_000:80_000] = 1
        b = Array(z.store, path=z.path, read_only=True, partial_decompress=True)
        assert (b[40_000:80_000] == 1).all()

    def test_read_from_all_blocks(self):
        '''Tests to make sure `PartialReadBuffer.read_part` doesn't fail when
        stop isn't in the `start_points` array
        '''
        z = self.create_array(shape=1000000, chunks=100_000)
        z[2:99_000] = 1
        b = Array(z.store, path=z.path, read_only=True, partial_decompress=True)
        assert (b[2:99_000] == 1).all()


@pytest.mark.skipif(have_fsspec is False, reason="needs fsspec")
class TestArrayWithFSStoreV3Nested(TestArrayWithPathV3):

    @staticmethod
    def create_array(array_path='arr1', read_only=False, **kwargs):
        path = mkdtemp()
        atexit.register(shutil.rmtree, path)
        key_separator = kwargs.pop('key_separator', "/")
        store = FSStoreV3(path, key_separator=key_separator, auto_mkdir=True)
        cache_metadata = kwargs.pop('cache_metadata', True)
        cache_attrs = kwargs.pop('cache_attrs', True)
        kwargs.setdefault('compressor', Blosc())
        init_array(store, path=array_path, **kwargs)
        return Array(store, path=array_path, read_only=read_only, cache_metadata=cache_metadata,
                     cache_attrs=cache_attrs)

    def expected(self):
        return [
           "d8c6501914fd9e999a437c304cfe8c8f4a47f3a1",
           "b52ac4f6f1982611d9d4b9e32f027603f7076e15",
           "936b531463d9a7575c5766cb5fe6e9a85598fc4e",
           "b9fc48400b71f30de731dcd838b49d1bdbe3b21b",
           "dccb51bd549886fb676c255c99f78a437a5b6cd8",
        ]

    def test_hexdigest(self):
        found = []

        # Check basic 1-D array
        z = self.create_array(shape=(1050,), chunks=100, dtype='<i4')
        found.append(z.hexdigest())

        # Check basic 1-D array with different type
        z = self.create_array(shape=(1050,), chunks=100, dtype='<f4')
        found.append(z.hexdigest())

        # Check basic 2-D array
        z = self.create_array(shape=(20, 35,), chunks=10, dtype='<i4')
        found.append(z.hexdigest())

        # Check basic 1-D array with some data
        z = self.create_array(shape=(1050,), chunks=100, dtype='<i4')
        z[200:400] = np.arange(200, 400, dtype='i4')
        found.append(z.hexdigest())

        # Check basic 1-D array with attributes
        z = self.create_array(shape=(1050,), chunks=100, dtype='<i4')
        z.attrs['foo'] = 'bar'
        found.append(z.hexdigest())

        assert self.expected() == found


@pytest.mark.skipif(have_fsspec is False, reason="needs fsspec")
class TestArrayWithFSStoreV3NestedPartialRead(TestArrayWithPathV3):
    @staticmethod
    def create_array(array_path='arr1', read_only=False, **kwargs):
        path = mkdtemp()
        atexit.register(shutil.rmtree, path)
        key_separator = kwargs.pop('key_separator', "/")
        store = FSStoreV3(path, key_separator=key_separator, auto_mkdir=True)
        cache_metadata = kwargs.pop("cache_metadata", True)
        cache_attrs = kwargs.pop("cache_attrs", True)
        kwargs.setdefault("compressor", Blosc())
        init_array(store, path=array_path, **kwargs)
        return Array(
            store,
            path=array_path,
            read_only=read_only,
            cache_metadata=cache_metadata,
            cache_attrs=cache_attrs,
            partial_decompress=True,
        )

    def expected(self):
        return [
           "d8c6501914fd9e999a437c304cfe8c8f4a47f3a1",
           "b52ac4f6f1982611d9d4b9e32f027603f7076e15",
           "936b531463d9a7575c5766cb5fe6e9a85598fc4e",
           "b9fc48400b71f30de731dcd838b49d1bdbe3b21b",
           "dccb51bd549886fb676c255c99f78a437a5b6cd8",
        ]

    def test_hexdigest(self):
        found = []

        # Check basic 1-D array
        z = self.create_array(shape=(1050,), chunks=100, dtype="<i4")
        found.append(z.hexdigest())

        # Check basic 1-D array with different type
        z = self.create_array(shape=(1050,), chunks=100, dtype="<f4")
        found.append(z.hexdigest())

        # Check basic 2-D array
        z = self.create_array(
            shape=(
                20,
                35,
            ),
            chunks=10,
            dtype="<i4",
        )
        found.append(z.hexdigest())

        # Check basic 1-D array with some data
        z = self.create_array(shape=(1050,), chunks=100, dtype="<i4")
        z[200:400] = np.arange(200, 400, dtype="i4")
        found.append(z.hexdigest())

        # Check basic 1-D array with attributes
        z = self.create_array(shape=(1050,), chunks=100, dtype="<i4")
        z.attrs["foo"] = "bar"
        found.append(z.hexdigest())

        assert self.expected() == found

    def test_non_cont(self):
        z = self.create_array(shape=(500, 500, 500), chunks=(50, 50, 50), dtype="<i4")
        z[:, :, :] = 1
        # actually go through the partial read by accessing a single item
        assert z[0, :, 0].any()

    def test_read_nitems_less_than_blocksize_from_multiple_chunks(self):
        '''Tests to make sure decompression doesn't fail when `nitems` is
        less than a compressed block size, but covers multiple blocks
        '''
        z = self.create_array(shape=1000000, chunks=100_000)
        z[40_000:80_000] = 1
        b = Array(z.store, path=z.path, read_only=True, partial_decompress=True)
        assert (b[40_000:80_000] == 1).all()

    def test_read_from_all_blocks(self):
        '''Tests to make sure `PartialReadBuffer.read_part` doesn't fail when
        stop isn't in the `start_points` array
        '''
        z = self.create_array(shape=1000000, chunks=100_000)
        z[2:99_000] = 1
        b = Array(z.store, path=z.path, read_only=True, partial_decompress=True)
        assert (b[2:99_000] == 1).all()
