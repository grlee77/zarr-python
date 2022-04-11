"""This module contains storage classes related to redis"""

from numcodecs.compat import ensure_bytes
from zarr._storage.store import Store
from zarr._storage.store_v3 import RmdirV3, StoreV3

__doctest_requires__ = {
    ('RedisStore', 'RedisStore.*'): ['redis'],
    ('RedisStoreV3', 'RedisStoreV3.*'): ['redis'],
}


class RedisStore(Store):
    """Storage class using Redis.

    .. note:: This is an experimental feature.

    Requires the `redis <https://redis-py.readthedocs.io/>`_
    package to be installed.

    Parameters
    ----------
    prefix : string
        Name of prefix for Redis keys
    dimension_separator : {'.', '/'}, optional
        Separator placed between the dimensions of a chunk.
    **kwargs
        Keyword arguments passed through to the `redis.Redis` function.

    """
    def __init__(self, prefix='zarr', dimension_separator=None, **kwargs):
        import redis
        self._prefix = prefix
        self._kwargs = kwargs
        self._dimension_separator = dimension_separator

        self.client = redis.Redis(**kwargs)

    def _key(self, key):
        return '{prefix}:{key}'.format(prefix=self._prefix, key=key)

    def __getitem__(self, key):
        return self.client[self._key(key)]

    def __setitem__(self, key, value):
        value = ensure_bytes(value)
        self.client[self._key(key)] = value

    def __delitem__(self, key):
        count = self.client.delete(self._key(key))
        if not count:
            raise KeyError(key)

    def keylist(self):
        offset = len(self._key(''))  # length of prefix
        return [key[offset:].decode('utf-8')
                for key in self.client.keys(self._key('*'))]

    def keys(self):
        for key in self.keylist():
            yield key

    def __iter__(self):
        for key in self.keys():
            yield key

    def __len__(self):
        return len(self.keylist())

    def __getstate__(self):
        return self._prefix, self._kwargs

    def __setstate__(self, state):
        prefix, kwargs = state
        self.__init__(prefix=prefix, **kwargs)

    def clear(self):
        for key in self.keys():
            del self[key]


class RedisStoreV3(RmdirV3, RedisStore, StoreV3):

    def list(self):
        return list(self.keys())

    def __setitem__(self, key, value):
        self._validate_key(key)
        super().__setitem__(key, value)


RedisStoreV3.__doc__ = RedisStore.__doc__
