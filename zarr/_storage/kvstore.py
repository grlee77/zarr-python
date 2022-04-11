"""This module contains storage classes related to dict-like key-value stores"""

from zarr._storage.store import Store
from zarr._storage.store_v3 import RmdirV3, StoreV3


class KVStore(Store):
    """
    This provides a default implementation of a store interface around
    a mutable mapping, to avoid having to test stores for presence of methods.

    This, for most methods should just be a pass-through to the underlying KV
    store which is likely to expose a MutableMapping interface,
    """

    def __init__(self, mutablemapping):
        self._mutable_mapping = mutablemapping

    def __getitem__(self, key):
        return self._mutable_mapping[key]

    def __setitem__(self, key, value):
        self._mutable_mapping[key] = value

    def __delitem__(self, key):
        del self._mutable_mapping[key]

    def get(self, key, default=None):
        return self._mutable_mapping.get(key, default)

    def values(self):
        return self._mutable_mapping.values()

    def __iter__(self):
        return iter(self._mutable_mapping)

    def __len__(self):
        return len(self._mutable_mapping)

    def __repr__(self):
        return f"<{self.__class__.__name__}: \n{repr(self._mutable_mapping)}\n at {hex(id(self))}>"

    def __eq__(self, other):
        if isinstance(other, KVStore):
            return self._mutable_mapping == other._mutable_mapping
        else:
            return NotImplemented


class KVStoreV3(RmdirV3, KVStore, StoreV3):

    def list(self):
        return list(self._mutable_mapping.keys())

    def __setitem__(self, key, value):
        self._validate_key(key)
        super().__setitem__(key, value)

    def __eq__(self, other):
        return (
            isinstance(other, KVStoreV3) and
            self._mutable_mapping == other._mutable_mapping
        )


KVStoreV3.__doc__ = KVStore.__doc__
