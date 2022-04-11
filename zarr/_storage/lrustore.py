"""This module contains storage classes using LMDB"""
from collections import OrderedDict
from threading import Lock
from typing import Any, Dict

from zarr._storage.store import BaseStore, Path, Store, StoreLike, getsize, listdir
from zarr._storage.store_v3 import RmdirV3, StoreV3
from zarr.util import buffer_size


__doctest_requires__ = {
    ('LRUStoreCache', 'LRUStoreCache.*'): ['s3fs'],
}


class LRUStoreCache(Store):
    """Storage class that implements a least-recently-used (LRU) cache layer over
    some other store. Intended primarily for use with stores that can be slow to
    access, e.g., remote stores that require network communication to store and
    retrieve data.

    Parameters
    ----------
    store : Store
        The store containing the actual data to be cached.
    max_size : int
        The maximum size that the cache may grow to, in number of bytes. Provide `None`
        if you would like the cache to have unlimited size.

    Examples
    --------
    The example below wraps an S3 store with an LRU cache::

        >>> import s3fs
        >>> import zarr
        >>> s3 = s3fs.S3FileSystem(anon=True, client_kwargs=dict(region_name='eu-west-2'))
        >>> store = s3fs.S3Map(root='zarr-demo/store', s3=s3, check=False)
        >>> cache = zarr.LRUStoreCache(store, max_size=2**28)
        >>> root = zarr.group(store=cache)  # doctest: +REMOTE_DATA
        >>> z = root['foo/bar/baz']  # doctest: +REMOTE_DATA
        >>> from timeit import timeit
        >>> # first data access is relatively slow, retrieved from store
        ... timeit('print(z[:].tobytes())', number=1, globals=globals())  # doctest: +SKIP
        b'Hello from the cloud!'
        0.1081731989979744
        >>> # second data access is faster, uses cache
        ... timeit('print(z[:].tobytes())', number=1, globals=globals())  # doctest: +SKIP
        b'Hello from the cloud!'
        0.0009490990014455747

    """

    def __init__(self, store: StoreLike, max_size: int):
        self._store: BaseStore = BaseStore._ensure_store(store)
        self._max_size = max_size
        self._current_size = 0
        self._keys_cache = None
        self._contains_cache = None
        self._listdir_cache: Dict[Path, Any] = dict()
        self._values_cache: Dict[Path, Any] = OrderedDict()
        self._mutex = Lock()
        self.hits = self.misses = 0

    def __getstate__(self):
        return (self._store, self._max_size, self._current_size, self._keys_cache,
                self._contains_cache, self._listdir_cache, self._values_cache, self.hits,
                self.misses)

    def __setstate__(self, state):
        (self._store, self._max_size, self._current_size, self._keys_cache,
         self._contains_cache, self._listdir_cache, self._values_cache, self.hits,
         self.misses) = state
        self._mutex = Lock()

    def __len__(self):
        return len(self._keys())

    def __iter__(self):
        return self.keys()

    def __contains__(self, key):
        with self._mutex:
            if self._contains_cache is None:
                self._contains_cache = set(self._keys())
            return key in self._contains_cache

    def clear(self):
        self._store.clear()
        self.invalidate()

    def keys(self):
        with self._mutex:
            return iter(self._keys())

    def _keys(self):
        if self._keys_cache is None:
            self._keys_cache = list(self._store.keys())
        return self._keys_cache

    def listdir(self, path: Path = None):
        with self._mutex:
            try:
                return self._listdir_cache[path]
            except KeyError:
                listing = listdir(self._store, path)
                self._listdir_cache[path] = listing
                return listing

    def getsize(self, path=None) -> int:
        return getsize(self._store, path=path)

    def _pop_value(self):
        # remove the first value from the cache, as this will be the least recently
        # used value
        _, v = self._values_cache.popitem(last=False)
        return v

    def _accommodate_value(self, value_size):
        if self._max_size is None:
            return
        # ensure there is enough space in the cache for a new value
        while self._current_size + value_size > self._max_size:
            v = self._pop_value()
            self._current_size -= buffer_size(v)

    def _cache_value(self, key: Path, value):
        # cache a value
        value_size = buffer_size(value)
        # check size of the value against max size, as if the value itself exceeds max
        # size then we are never going to cache it
        if self._max_size is None or value_size <= self._max_size:
            self._accommodate_value(value_size)
            self._values_cache[key] = value
            self._current_size += value_size

    def invalidate(self):
        """Completely clear the cache."""
        with self._mutex:
            self._values_cache.clear()
            self._invalidate_keys()

    def invalidate_values(self):
        """Clear the values cache."""
        with self._mutex:
            self._values_cache.clear()

    def invalidate_keys(self):
        """Clear the keys cache."""
        with self._mutex:
            self._invalidate_keys()

    def _invalidate_keys(self):
        self._keys_cache = None
        self._contains_cache = None
        self._listdir_cache.clear()

    def _invalidate_value(self, key):
        if key in self._values_cache:
            value = self._values_cache.pop(key)
            self._current_size -= buffer_size(value)

    def __getitem__(self, key):
        try:
            # first try to obtain the value from the cache
            with self._mutex:
                value = self._values_cache[key]
                # cache hit if no KeyError is raised
                self.hits += 1
                # treat the end as most recently used
                self._values_cache.move_to_end(key)

        except KeyError:
            # cache miss, retrieve value from the store
            value = self._store[key]
            with self._mutex:
                self.misses += 1
                # need to check if key is not in the cache, as it may have been cached
                # while we were retrieving the value from the store
                if key not in self._values_cache:
                    self._cache_value(key, value)

        return value

    def __setitem__(self, key, value):
        self._store[key] = value
        with self._mutex:
            self._invalidate_keys()
            self._invalidate_value(key)
            self._cache_value(key, value)

    def __delitem__(self, key):
        del self._store[key]
        with self._mutex:
            self._invalidate_keys()
            self._invalidate_value(key)


class LRUStoreCacheV3(RmdirV3, LRUStoreCache, StoreV3):

    def __init__(self, store, max_size: int):
        self._store = StoreV3._ensure_store(store)
        self._max_size = max_size
        self._current_size = 0
        self._keys_cache = None
        self._contains_cache = None
        self._listdir_cache: Dict[Path, Any] = dict()
        self._values_cache: Dict[Path, Any] = OrderedDict()
        self._mutex = Lock()
        self.hits = self.misses = 0

    def list(self):
        return list(self.keys())

    def __setitem__(self, key, value):
        self._validate_key(key)
        super().__setitem__(key, value)


LRUStoreCacheV3.__doc__ = LRUStoreCache.__doc__
