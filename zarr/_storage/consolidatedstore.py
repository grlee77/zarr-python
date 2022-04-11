"""This module contains storage classes for consolidated metadata storage"""

from zarr._storage.kvstore import KVStore, KVStoreV3
from zarr._storage.store import Store, StoreLike, getsize, listdir
from zarr._storage.store_v3 import meta_root, StoreV3
from zarr.errors import MetadataError, ReadOnlyError
from zarr.util import json_loads


class ConsolidatedMetadataStore(Store):
    """A layer over other storage, where the metadata has been consolidated into
    a single key.

    The purpose of this class, is to be able to get all of the metadata for
    a given array in a single read operation from the underlying storage.
    See :func:`zarr.convenience.consolidate_metadata` for how to create this
    single metadata key.

    This class loads from the one key, and stores the data in a dict, so that
    accessing the keys no longer requires operations on the backend store.

    This class is read-only, and attempts to change the array metadata will
    fail, but changing the data is possible. If the backend storage is changed
    directly, then the metadata stored here could become obsolete, and
    :func:`zarr.convenience.consolidate_metadata` should be called again and the class
    re-invoked. The use case is for write once, read many times.

    .. versionadded:: 2.3

    .. note:: This is an experimental feature.

    Parameters
    ----------
    store: Store
        Containing the zarr array.
    metadata_key: str
        The target in the store where all of the metadata are stored. We
        assume JSON encoding.

    See Also
    --------
    zarr.convenience.consolidate_metadata, zarr.convenience.open_consolidated

    """

    def __init__(self, store: StoreLike, metadata_key=".zmetadata"):
        self.store = Store._ensure_store(store)

        # retrieve consolidated metadata
        meta = json_loads(self.store[metadata_key])

        # check format of consolidated metadata
        consolidated_format = meta.get('zarr_consolidated_format', None)
        if consolidated_format != 1:
            raise MetadataError('unsupported zarr consolidated metadata format: %s' %
                                consolidated_format)

        # decode metadata
        self.meta_store: Store = KVStore(meta["metadata"])

    def __getitem__(self, key):
        return self.meta_store[key]

    def __contains__(self, item):
        return item in self.meta_store

    def __iter__(self):
        return iter(self.meta_store)

    def __len__(self):
        return len(self.meta_store)

    def __delitem__(self, key):
        raise ReadOnlyError()

    def __setitem__(self, key, value):
        raise ReadOnlyError()

    def getsize(self, path):
        return getsize(self.meta_store, path)

    def listdir(self, path):
        return listdir(self.meta_store, path)


class ConsolidatedMetadataStoreV3(ConsolidatedMetadataStore, StoreV3):
    """A layer over other storage, where the metadata has been consolidated into
    a single key.

    The purpose of this class, is to be able to get all of the metadata for
    a given array in a single read operation from the underlying storage.
    See :func:`zarr.convenience.consolidate_metadata` for how to create this
    single metadata key.

    This class loads from the one key, and stores the data in a dict, so that
    accessing the keys no longer requires operations on the backend store.

    This class is read-only, and attempts to change the array metadata will
    fail, but changing the data is possible. If the backend storage is changed
    directly, then the metadata stored here could become obsolete, and
    :func:`zarr.convenience.consolidate_metadata` should be called again and the class
    re-invoked. The use case is for write once, read many times.

    .. note:: This is an experimental feature.

    Parameters
    ----------
    store: Store
        Containing the zarr array.
    metadata_key: str
        The target in the store where all of the metadata are stored. We
        assume JSON encoding.

    See Also
    --------
    zarr.convenience.consolidate_metadata, zarr.convenience.open_consolidated

    """

    def __init__(self, store: StoreLike, metadata_key=meta_root + "consolidated/.zmetadata"):
        self.store = StoreV3._ensure_store(store)

        # retrieve consolidated metadata
        meta = json_loads(self.store[metadata_key])

        # check format of consolidated metadata
        consolidated_format = meta.get('zarr_consolidated_format', None)
        if consolidated_format != 1:
            raise MetadataError('unsupported zarr consolidated metadata format: %s' %
                                consolidated_format)

        # decode metadata
        self.meta_store: Store = KVStoreV3(meta["metadata"])

    def rmdir(self, key):
        raise ReadOnlyError()
