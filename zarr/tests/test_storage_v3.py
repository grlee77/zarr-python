import array
import atexit
import os
import tempfile

import numpy as np
import pytest

from numcodecs.compat import ensure_bytes

from zarr.codecs import Zlib
from zarr.errors import ContainsArrayError, ContainsGroupError
from zarr.meta import ZARR_FORMAT, ZARR_FORMAT_v3
from zarr.storage import (array_meta_key, atexit_rmglob, atexit_rmtree,
                          default_compressor, getsize, init_array, init_group)
from zarr.storage import (KVStoreV3, MemoryStoreV3, ZipStoreV3, FSStoreV3,
                          DirectoryStoreV3, NestedDirectoryStoreV3,
                          RedisStoreV3, MongoDBStoreV3, DBMStoreV3,
                          LMDBStoreV3, SQLiteStoreV3, LRUStoreCacheV3)
from zarr.tests.util import CountingDictV3, have_fsspec, skip_test_env_var

from .test_storage import (StoreTests, TestMemoryStore, TestDirectoryStore,
                           TestFSStore, TestNestedDirectoryStore, TestZipStore,
                           TestDBMStore, TestDBMStoreDumb, TestDBMStoreGnu,
                           TestDBMStoreNDBM, TestDBMStoreBerkeleyDB,
                           TestLMDBStore, TestSQLiteStore,
                           TestSQLiteStoreInMemory, TestLRUStoreCache,
                           dimension_separator_fixture, s3,
                           skip_if_nested_chunks)


@pytest.fixture(params=[
    (None, "/"),
    (".", "."),
    ("/", "/"),
])
def dimension_separator_fixture_v3(request):
    return request.param


class StoreV3Tests(StoreTests):

    def test_getsize(self):
        # TODO: determine proper getsize() behavior for v3

        # Currently returns the combined size of entries under
        # meta/root/path and data/root/path.
        # Any path not under meta/root/ or data/root/ (including zarr.json)
        # returns size 0.

        store = self.create_store()
        if isinstance(store, dict) or hasattr(store, 'getsize'):
            assert 0 == getsize(store, 'zarr.json')
            store['meta/root/foo/a'] = b'x'
            assert 1 == getsize(store)
            assert 1 == getsize(store, 'foo')
            store['meta/root/foo/b'] = b'x'
            assert 2 == getsize(store, 'foo')
            assert 1 == getsize(store, 'foo/b')
            store['meta/root/bar/a'] = b'yy'
            assert 2 == getsize(store, 'bar')
            store['data/root/bar/a'] = b'zzz'
            assert 5 == getsize(store, 'bar')
            store['data/root/baz/a'] = b'zzz'
            assert 3 == getsize(store, 'baz')
            assert 10 == getsize(store)
            store['data/root/quux'] = array.array('B', b'zzzz')
            assert 14 == getsize(store)
            assert 4 == getsize(store, 'quux')
            store['data/root/spong'] = np.frombuffer(b'zzzzz', dtype='u1')
            assert 19 == getsize(store)
            assert 5 == getsize(store, 'spong')

        store.close()

    # noinspection PyStatementEffect
    def test_hierarchy(self):
        pytest.skip("TODO: adapt v2 test_hierarchy tests to v3")

    def test_init_array(self, dimension_separator_fixture_v3):

        pass_dim_sep, want_dim_sep = dimension_separator_fixture_v3

        store = self.create_store()
        path = 'arr1'
        init_array(store, path=path, shape=1000, chunks=100,
                   dimension_separator=pass_dim_sep)

        # check metadata
        mkey = 'meta/root/' + path + '.array.json'
        assert mkey in store
        meta = store._metadata_class.decode_array_metadata(store[mkey])
        # TODO: zarr_format already stored at the heirarchy level should we
        #       also keep it in the .array.json?
        assert ZARR_FORMAT_v3 == meta['zarr_format']
        assert (1000,) == meta['shape']
        assert (100,) == meta['chunk_grid']['chunk_shape']
        assert np.dtype(None) == meta['data_type']
        assert default_compressor.get_config() == meta['compressor']
        assert meta['fill_value'] is None
        # Missing MUST be assumed to be "/"
        assert meta.get('dimension_separator', "/") is want_dim_sep
        assert meta['chunk_grid']['separator'] is want_dim_sep
        store.close()

    def _test_init_array_overwrite(self, order):
        # setup
        store = self.create_store()

        if store._store_version < 3:
            path = None
            mkey = array_meta_key
        else:
            path = 'arr1'  # no default, have to specify for v3
            mkey = 'meta/root/' + path + '.array.json'
        store[mkey] = store._metadata_class.encode_array_metadata(
            dict(shape=(2000,),
                 chunk_grid=dict(type='regular',
                                 chunk_shape=(200,),
                                 separator=('/')),
                 data_type=np.dtype('u1'),
                 compressor=Zlib(1).get_config(),
                 fill_value=0,
                 chunk_memory_layout=order,
                 filters=None)
        )

        # don't overwrite (default)
        with pytest.raises(ContainsArrayError):
            init_array(store, path=path, shape=1000, chunks=100)

        # do overwrite
        try:
            init_array(store, path=path, shape=1000, chunks=100,
                       dtype='i4', overwrite=True)
        except NotImplementedError:
            pass
        else:
            assert mkey in store
            meta = store._metadata_class.decode_array_metadata(
                store[mkey]
            )
            assert (1000,) == meta['shape']
            if store._store_version == 2:
                assert ZARR_FORMAT == meta['zarr_format']
                assert (100,) == meta['chunks']
                assert np.dtype('i4') == meta['dtype']
            elif store._store_version == 3:
                assert ZARR_FORMAT_v3 == meta['zarr_format']
                assert (100,) == meta['chunk_grid']['chunk_shape']
                assert np.dtype('i4') == meta['data_type']
            else:
                raise ValueError(
                    "unexpected store version: {store._store_version}"
                )
        store.close()

    def test_init_array_path(self):
        path = 'foo/bar'
        store = self.create_store()
        init_array(store, shape=1000, chunks=100, path=path)

        # check metadata
        mkey = 'meta/root/' + path + '.array.json'
        assert mkey in store
        meta = store._metadata_class.decode_array_metadata(store[mkey])
        assert ZARR_FORMAT_v3 == meta['zarr_format']
        assert (1000,) == meta['shape']
        assert (100,) == meta['chunk_grid']['chunk_shape']
        assert np.dtype(None) == meta['data_type']
        assert default_compressor.get_config() == meta['compressor']
        assert meta['fill_value'] is None

        store.close()

    def _test_init_array_overwrite_path(self, order):
        # setup
        path = 'foo/bar'
        store = self.create_store()
        meta = dict(shape=(2000,),
                    chunk_grid=dict(type='regular',
                                    chunk_shape=(200,),
                                    separator=('/')),
                    data_type=np.dtype('u1'),
                    compressor=Zlib(1).get_config(),
                    fill_value=0,
                    chunk_memory_layout=order,
                    filters=None)
        mkey = 'meta/root/' + path + '.array.json'
        store[mkey] = store._metadata_class.encode_array_metadata(meta)

        # don't overwrite
        with pytest.raises(ContainsArrayError):
            init_array(store, shape=1000, chunks=100, path=path)

        # do overwrite
        try:
            init_array(store, shape=1000, chunks=100, dtype='i4', path=path,
                       overwrite=True)
        except NotImplementedError:
            pass
        else:
            assert mkey in store
            # should have been overwritten
            meta = store._metadata_class.decode_array_metadata(store[mkey])
            assert ZARR_FORMAT_v3 == meta['zarr_format']
            assert (1000,) == meta['shape']
            assert (100,) == meta['chunk_grid']['chunk_shape']
            assert np.dtype('i4') == meta['data_type']

        store.close()

    def test_init_array_overwrite_group(self):
        # setup
        path = 'foo/bar'
        store = self.create_store()
        array_key = 'meta/root/' + path + '.array.json'
        group_key = 'meta/root/' + path + '.group.json'
        store[group_key] = store._metadata_class.encode_group_metadata()

        with pytest.raises(ContainsGroupError):
            init_array(store, shape=1000, chunks=100, path=path)

        # do overwrite
        try:
            init_array(store, shape=1000, chunks=100, dtype='i4', path=path,
                       overwrite=True)
        except NotImplementedError:
            pass
        else:
            assert group_key not in store
            assert array_key in store
            meta = store._metadata_class.decode_array_metadata(
                store[array_key]
            )
            assert ZARR_FORMAT_v3 == meta['zarr_format']
            assert (1000,) == meta['shape']
            assert (100,) == meta['chunk_grid']['chunk_shape']
            assert np.dtype('i4') == meta['data_type']

        store.close()

    def _test_init_array_overwrite_chunk_store(self, order):
        # setup
        store = self.create_store()
        chunk_store = self.create_store()
        path = 'arr1'
        mkey = 'meta/root/' + path + '.array.json'
        store[mkey] = store._metadata_class.encode_array_metadata(
            dict(shape=(2000,),
                 chunk_grid=dict(type='regular',
                                 chunk_shape=(200,),
                                 separator=('/')),
                 data_type=np.dtype('u1'),
                 compressor=None,
                 fill_value=0,
                 filters=None,
                 chunk_memory_layout=order)
        )

        chunk_store['data/root/arr1/0'] = b'aaa'
        chunk_store['data/root/arr1/1'] = b'bbb'

        assert 'data/root/arr1/0' in chunk_store
        assert 'data/root/arr1/1' in chunk_store

        # don't overwrite (default)
        with pytest.raises(ValueError):
            init_array(store, path=path, shape=1000, chunks=100, chunk_store=chunk_store)

        # do overwrite
        try:
            init_array(store, path=path, shape=1000, chunks=100, dtype='i4',
                       overwrite=True, chunk_store=chunk_store)
        except NotImplementedError:
            pass
        else:
            assert mkey in store
            meta = store._metadata_class.decode_array_metadata(store[mkey])
            assert ZARR_FORMAT_v3 == meta['zarr_format']
            assert (1000,) == meta['shape']
            assert (100,) == meta['chunk_grid']['chunk_shape']
            assert np.dtype('i4') == meta['data_type']
            assert 'data/root/arr1/0' not in chunk_store
            assert 'data/root/arr1/1' not in chunk_store

        store.close()
        chunk_store.close()

    def test_init_array_compat(self):
        store = self.create_store()
        path = 'arr1'
        init_array(store, path=path, shape=1000, chunks=100, compressor='none')
        mkey = 'meta/root/' + path + '.array.json'
        meta = store._metadata_class.decode_array_metadata(
            store[mkey]
        )
        assert meta['compressor'] is None

        store.close()

    def test_init_group(self):
        store = self.create_store()
        path = "meta/root/foo"
        init_group(store, path=path)

        # check metadata
        mkey = 'meta/root/' + path + '.group.json'
        assert mkey in store
        meta = store._metadata_class.decode_group_metadata(store[mkey])
        assert meta == {'attributes': {}}

        store.close()

    def _test_init_group_overwrite(self, order):
        pytest.skip(
            "In v3 array and group names cannot overlap"
        )

    def _test_init_group_overwrite_path(self, order):
        # setup
        path = 'foo/bar'
        store = self.create_store()
        meta = dict(
            shape=(2000,),
            chunk_grid=dict(type='regular',
                            chunk_shape=(200,),
                            separator=('/')),
            data_type=np.dtype('u1'),
            compressor=None,
            fill_value=0,
            filters=None,
            chunk_memory_layout=order,
        )
        array_key = 'meta/root/' + path + '.array.json'
        group_key = 'meta/root/' + path + '.group.json'
        store[array_key] = store._metadata_class.encode_array_metadata(meta)

        # don't overwrite
        with pytest.raises(ContainsArrayError):
            init_group(store, path=path)

        # do overwrite
        try:
            init_group(store, overwrite=True, path=path)
        except NotImplementedError:
            pass
        else:
            assert array_key not in store
            assert group_key in store
            # should have been overwritten
            meta = store._metadata_class.decode_group_metadata(store[group_key])
            # assert ZARR_FORMAT == meta['zarr_format']
            assert meta == {'attributes': {}}

        store.close()

    def _test_init_group_overwrite_chunk_store(self, order):
        pytest.skip(
            "In v3 array and group names cannot overlap"
        )


class TestMappingStoreV3(StoreV3Tests):

    def create_store(self, **kwargs):
        return KVStoreV3(dict())

    def test_set_invalid_content(self):
        # Generic mappings support non-buffer types
        pass


class TestMemoryStoreV3(TestMemoryStore, StoreV3Tests):

    def create_store(self, **kwargs):
        skip_if_nested_chunks(**kwargs)
        return MemoryStoreV3(**kwargs)


class TestDirectoryStoreV3(TestDirectoryStore, StoreV3Tests):

    def create_store(self, normalize_keys=False, **kwargs):
        # For v3, don't have to skip if nested.
        # skip_if_nested_chunks(**kwargs)

        path = tempfile.mkdtemp()
        atexit.register(atexit_rmtree, path)
        store = DirectoryStoreV3(path, normalize_keys=normalize_keys, **kwargs)
        return store


@pytest.mark.skipif(have_fsspec is False, reason="needs fsspec")
class TestFSStoreV3(TestFSStore, StoreV3Tests):

    def create_store(self, normalize_keys=False,
                     dimension_separator=".",
                     path=None,
                     **kwargs):

        if path is None:
            path = tempfile.mkdtemp()
            atexit.register(atexit_rmtree, path)

        store = FSStoreV3(
            path,
            normalize_keys=normalize_keys,
            dimension_separator=dimension_separator,
            **kwargs)
        return store

    def test_init_array(self):
        store = self.create_store()
        path = 'arr1'
        init_array(store, path=path, shape=1000, chunks=100)

        # check metadata
        array_meta_key = 'meta/root/' + path + '.array.json'
        assert array_meta_key in store
        meta = store._metadata_class.decode_array_metadata(store[array_meta_key])
        assert ZARR_FORMAT_v3 == meta['zarr_format']
        assert (1000,) == meta['shape']
        assert (100,) == meta['chunk_grid']['chunk_shape']
        assert np.dtype(None) == meta['data_type']
        assert meta['chunk_grid']['separator'] == "/"

    # TODO: remove this skip once v3 support is added to hierarchy.Group
    @pytest.mark.skipif(True, reason="need v3 support in zarr.hierarchy.Group")
    def test_deep_ndim(self):
        import zarr

        store = self.create_store()
        foo = zarr.open_group(store=store, path='group1')
        bar = foo.create_group("bar")
        baz = bar.create_dataset("baz",
                                 shape=(4, 4, 4),
                                 chunks=(2, 2, 2),
                                 dtype="i8")
        baz[:] = 1
        assert set(store.listdir()) == set(["data", "meta", "zarr.json"])
        assert set(store.listdir("meta/root/group1")) == set(["bar", "bar.group.json"])
        assert set(store.listdir("data/root/group1")) == set(["bar"])
        assert foo["bar"]["baz"][(0, 0, 0)] == 1


@pytest.mark.skipif(have_fsspec is False, reason="needs fsspec")
class TestFSStoreV3WithKeySeparator(StoreV3Tests):

    def create_store(self, normalize_keys=False, key_separator=".", **kwargs):

        # Since the user is passing key_separator, that will take priority.
        skip_if_nested_chunks(**kwargs)

        path = tempfile.mkdtemp()
        atexit.register(atexit_rmtree, path)
        return FSStoreV3(
            path,
            normalize_keys=normalize_keys,
            key_separator=key_separator)


# TODO: remove NestedDirectoryStoreV3?
class TestNestedDirectoryStoreV3(TestNestedDirectoryStore,
                                 TestDirectoryStoreV3):

    def create_store(self, normalize_keys=False, **kwargs):
        path = tempfile.mkdtemp()
        atexit.register(atexit_rmtree, path)
        store = NestedDirectoryStoreV3(path, normalize_keys=normalize_keys, **kwargs)
        return store

    def test_init_array(self):
        store = self.create_store()
        # assert store._dimension_separator == "/"
        path = 'arr1'
        init_array(store, path=path, shape=1000, chunks=100)

        # check metadata
        array_meta_key = 'meta/root/' + path + '.array.json'
        assert array_meta_key in store
        meta = store._metadata_class.decode_array_metadata(store[array_meta_key])
        assert ZARR_FORMAT_v3 == meta['zarr_format']
        assert (1000,) == meta['shape']
        assert (100,) == meta['chunk_grid']['chunk_shape']
        assert np.dtype(None) == meta['data_type']
        # assert meta['dimension_separator'] == "/"
        assert meta['chunk_grid']['separator'] == "/"

# TODO: enable once N5StoreV3 has been implemented
# @pytest.mark.skipif(True, reason="N5StoreV3 not yet fully implemented")
# class TestN5StoreV3(TestN5Store, TestNestedDirectoryStoreV3, StoreV3Tests):


class TestZipStoreV3(TestZipStore, StoreV3Tests):

    def create_store(self, **kwargs):
        path = tempfile.mktemp(suffix='.zip')
        atexit.register(os.remove, path)
        store = ZipStoreV3(path, mode='w', **kwargs)
        return store

    def test_mode(self):
        with ZipStoreV3('data/store.zip', mode='w') as store:
            store['foo'] = b'bar'
        store = ZipStoreV3('data/store.zip', mode='r')
        with pytest.raises(PermissionError):
            store['foo'] = b'bar'
        with pytest.raises(PermissionError):
            store.clear()


class TestDBMStoreV3(TestDBMStore, StoreV3Tests):

    def create_store(self, dimension_separator=None):
        path = tempfile.mktemp(suffix='.anydbm')
        atexit.register(atexit_rmglob, path + '*')
        # create store using default dbm implementation
        store = DBMStoreV3(path, flag='n', dimension_separator=dimension_separator)
        return store


class TestDBMStoreV3Dumb(TestDBMStoreDumb, StoreV3Tests):

    def create_store(self, **kwargs):
        path = tempfile.mktemp(suffix='.dumbdbm')
        atexit.register(atexit_rmglob, path + '*')

        import dbm.dumb as dumbdbm
        store = DBMStoreV3(path, flag='n', open=dumbdbm.open, **kwargs)
        return store


class TestDBMStoreV3Gnu(TestDBMStoreGnu, StoreV3Tests):

    def create_store(self, **kwargs):
        gdbm = pytest.importorskip("dbm.gnu")
        path = tempfile.mktemp(suffix=".gdbm")  # pragma: no cover
        atexit.register(os.remove, path)  # pragma: no cover
        store = DBMStoreV3(
            path, flag="n", open=gdbm.open, write_lock=False, **kwargs
        )  # pragma: no cover
        return store  # pragma: no cover


class TestDBMStoreV3NDBM(TestDBMStoreNDBM, StoreV3Tests):

    def create_store(self, **kwargs):
        ndbm = pytest.importorskip("dbm.ndbm")
        path = tempfile.mktemp(suffix=".ndbm")  # pragma: no cover
        atexit.register(atexit_rmglob, path + "*")  # pragma: no cover
        store = DBMStoreV3(path, flag="n", open=ndbm.open, **kwargs)  # pragma: no cover
        return store  # pragma: no cover


class TestDBMStoreV3BerkeleyDB(TestDBMStoreBerkeleyDB, StoreV3Tests):

    def create_store(self, **kwargs):
        bsddb3 = pytest.importorskip("bsddb3")
        path = tempfile.mktemp(suffix='.dbm')
        atexit.register(os.remove, path)
        store = DBMStoreV3(path, flag='n', open=bsddb3.btopen, write_lock=False, **kwargs)
        return store


class TestLMDBStoreV3(TestLMDBStore, StoreV3Tests):

    def create_store(self, **kwargs):
        pytest.importorskip("lmdb")
        path = tempfile.mktemp(suffix='.lmdb')
        atexit.register(atexit_rmtree, path)
        buffers = True
        store = LMDBStoreV3(path, buffers=buffers, **kwargs)
        return store


class TestSQLiteStoreV3(TestSQLiteStore, StoreV3Tests):

    def create_store(self, **kwargs):
        pytest.importorskip("sqlite3")
        path = tempfile.mktemp(suffix='.db')
        atexit.register(atexit_rmtree, path)
        store = SQLiteStoreV3(path, **kwargs)
        return store


class TestSQLiteStoreV3InMemory(TestSQLiteStoreInMemory, StoreV3Tests):

    def create_store(self, **kwargs):
        pytest.importorskip("sqlite3")
        store = SQLiteStoreV3(':memory:', **kwargs)
        return store


@skip_test_env_var("ZARR_TEST_MONGO")
class TestMongoDBStoreV3(StoreV3Tests):

    def create_store(self, **kwargs):
        pytest.importorskip("pymongo")
        store = MongoDBStoreV3(host='127.0.0.1', database='zarr_tests',
                               collection='zarr_tests', **kwargs)
        # start with an empty store
        store.clear()
        return store


@skip_test_env_var("ZARR_TEST_REDIS")
class TestRedisStoreV3(StoreV3Tests):

    def create_store(self, **kwargs):
        # TODO: this is the default host for Redis on Travis,
        # we probably want to generalize this though
        pytest.importorskip("redis")
        store = RedisStoreV3(host='localhost', port=6379, **kwargs)
        # start with an empty store
        store.clear()
        return store


class TestLRUStoreCacheV3(TestLRUStoreCache, StoreV3Tests):

    def create_store(self, **kwargs):
        # wrapper therefore no dimension_separator argument
        skip_if_nested_chunks(**kwargs)
        return LRUStoreCacheV3(dict(), max_size=2**27)

    def test_cache_values_no_max_size(self):

        # setup store
        store = CountingDictV3()
        store['foo'] = b'xxx'
        store['bar'] = b'yyy'
        assert 0 == store.counter['__getitem__', 'foo']
        assert 1 == store.counter['__setitem__', 'foo']
        assert 0 == store.counter['__getitem__', 'bar']
        assert 1 == store.counter['__setitem__', 'bar']

        # setup cache
        cache = LRUStoreCacheV3(store, max_size=None)
        assert 0 == cache.hits
        assert 0 == cache.misses

        # test first __getitem__, cache miss
        assert b'xxx' == cache['foo']
        assert 1 == store.counter['__getitem__', 'foo']
        assert 1 == store.counter['__setitem__', 'foo']
        assert 0 == cache.hits
        assert 1 == cache.misses

        # test second __getitem__, cache hit
        assert b'xxx' == cache['foo']
        assert 1 == store.counter['__getitem__', 'foo']
        assert 1 == store.counter['__setitem__', 'foo']
        assert 1 == cache.hits
        assert 1 == cache.misses

        # test __setitem__, __getitem__
        cache['foo'] = b'zzz'
        assert 1 == store.counter['__getitem__', 'foo']
        assert 2 == store.counter['__setitem__', 'foo']
        # should be a cache hit
        assert b'zzz' == cache['foo']
        assert 1 == store.counter['__getitem__', 'foo']
        assert 2 == store.counter['__setitem__', 'foo']
        assert 2 == cache.hits
        assert 1 == cache.misses

        # manually invalidate all cached values
        cache.invalidate_values()
        assert b'zzz' == cache['foo']
        assert 2 == store.counter['__getitem__', 'foo']
        assert 2 == store.counter['__setitem__', 'foo']
        cache.invalidate()
        assert b'zzz' == cache['foo']
        assert 3 == store.counter['__getitem__', 'foo']
        assert 2 == store.counter['__setitem__', 'foo']

        # test __delitem__
        del cache['foo']
        with pytest.raises(KeyError):
            # noinspection PyStatementEffect
            cache['foo']
        with pytest.raises(KeyError):
            # noinspection PyStatementEffect
            store['foo']

        # verify other keys untouched
        assert 0 == store.counter['__getitem__', 'bar']
        assert 1 == store.counter['__setitem__', 'bar']

    def test_cache_values_with_max_size(self):

        # setup store
        store = CountingDictV3()
        store['foo'] = b'xxx'
        store['bar'] = b'yyy'
        assert 0 == store.counter['__getitem__', 'foo']
        assert 0 == store.counter['__getitem__', 'bar']
        # setup cache - can only hold one item
        cache = LRUStoreCacheV3(store, max_size=5)
        assert 0 == cache.hits
        assert 0 == cache.misses

        # test first 'foo' __getitem__, cache miss
        assert b'xxx' == cache['foo']
        assert 1 == store.counter['__getitem__', 'foo']
        assert 0 == cache.hits
        assert 1 == cache.misses

        # test second 'foo' __getitem__, cache hit
        assert b'xxx' == cache['foo']
        assert 1 == store.counter['__getitem__', 'foo']
        assert 1 == cache.hits
        assert 1 == cache.misses

        # test first 'bar' __getitem__, cache miss
        assert b'yyy' == cache['bar']
        assert 1 == store.counter['__getitem__', 'bar']
        assert 1 == cache.hits
        assert 2 == cache.misses

        # test second 'bar' __getitem__, cache hit
        assert b'yyy' == cache['bar']
        assert 1 == store.counter['__getitem__', 'bar']
        assert 2 == cache.hits
        assert 2 == cache.misses

        # test 'foo' __getitem__, should have been evicted, cache miss
        assert b'xxx' == cache['foo']
        assert 2 == store.counter['__getitem__', 'foo']
        assert 2 == cache.hits
        assert 3 == cache.misses

        # test 'bar' __getitem__, should have been evicted, cache miss
        assert b'yyy' == cache['bar']
        assert 2 == store.counter['__getitem__', 'bar']
        assert 2 == cache.hits
        assert 4 == cache.misses

        # setup store
        store = CountingDictV3()
        store['foo'] = b'xxx'
        store['bar'] = b'yyy'
        assert 0 == store.counter['__getitem__', 'foo']
        assert 0 == store.counter['__getitem__', 'bar']
        # setup cache - can hold two items
        cache = LRUStoreCacheV3(store, max_size=6)
        assert 0 == cache.hits
        assert 0 == cache.misses

        # test first 'foo' __getitem__, cache miss
        assert b'xxx' == cache['foo']
        assert 1 == store.counter['__getitem__', 'foo']
        assert 0 == cache.hits
        assert 1 == cache.misses

        # test second 'foo' __getitem__, cache hit
        assert b'xxx' == cache['foo']
        assert 1 == store.counter['__getitem__', 'foo']
        assert 1 == cache.hits
        assert 1 == cache.misses

        # test first 'bar' __getitem__, cache miss
        assert b'yyy' == cache['bar']
        assert 1 == store.counter['__getitem__', 'bar']
        assert 1 == cache.hits
        assert 2 == cache.misses

        # test second 'bar' __getitem__, cache hit
        assert b'yyy' == cache['bar']
        assert 1 == store.counter['__getitem__', 'bar']
        assert 2 == cache.hits
        assert 2 == cache.misses

        # test 'foo' __getitem__, should still be cached
        assert b'xxx' == cache['foo']
        assert 1 == store.counter['__getitem__', 'foo']
        assert 3 == cache.hits
        assert 2 == cache.misses

        # test 'bar' __getitem__, should still be cached
        assert b'yyy' == cache['bar']
        assert 1 == store.counter['__getitem__', 'bar']
        assert 4 == cache.hits
        assert 2 == cache.misses

    def test_cache_keys(self):

        # setup
        store = CountingDictV3()
        store['foo'] = b'xxx'
        store['bar'] = b'yyy'
        assert 0 == store.counter['__contains__', 'foo']
        assert 0 == store.counter['__iter__']
        assert 0 == store.counter['keys']
        cache = LRUStoreCacheV3(store, max_size=None)

        # keys should be cached on first call
        keys = sorted(cache.keys())
        assert keys == ['bar', 'foo']
        assert 1 == store.counter['keys']
        # keys should now be cached
        assert keys == sorted(cache.keys())
        assert 1 == store.counter['keys']
        assert 'foo' in cache
        assert 0 == store.counter['__contains__', 'foo']
        assert keys == sorted(cache)
        assert 0 == store.counter['__iter__']
        assert 1 == store.counter['keys']

        # cache should be cleared if store is modified - crude but simple for now
        cache['baz'] = b'zzz'
        keys = sorted(cache.keys())
        assert keys == ['bar', 'baz', 'foo']
        assert 2 == store.counter['keys']
        # keys should now be cached
        assert keys == sorted(cache.keys())
        assert 2 == store.counter['keys']

        # manually invalidate keys
        cache.invalidate_keys()
        keys = sorted(cache.keys())
        assert keys == ['bar', 'baz', 'foo']
        assert 3 == store.counter['keys']
        assert 0 == store.counter['__contains__', 'foo']
        assert 0 == store.counter['__iter__']
        cache.invalidate_keys()
        keys = sorted(cache)
        assert keys == ['bar', 'baz', 'foo']
        assert 4 == store.counter['keys']
        assert 0 == store.counter['__contains__', 'foo']
        assert 0 == store.counter['__iter__']
        cache.invalidate_keys()
        assert 'foo' in cache
        assert 5 == store.counter['keys']
        assert 0 == store.counter['__contains__', 'foo']
        assert 0 == store.counter['__iter__']

        # check these would get counted if called directly
        assert 'foo' in store
        assert 1 == store.counter['__contains__', 'foo']
        assert keys == sorted(store)
        assert 1 == store.counter['__iter__']


# TODO: implement ABSStoreV3
# @skip_test_env_var("ZARR_TEST_ABS")
# class TestABSStoreV3(TestABSStore, StoreV3Tests):
