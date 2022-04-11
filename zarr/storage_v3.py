import os
from typing import Any

from zarr._storage.absstore import ABSStoreV3  # noqa: F401
from zarr._storage.consolidatedstore import ConsolidatedMetadataStoreV3  # noqa: F401
from zarr._storage.directorystore import DirectoryStoreV3  # noqa: F401
from zarr._storage.dbmstore import DBMStoreV3  # noqa: F401
from zarr._storage.fsstore import FSStoreV3  # noqa: F401
from zarr._storage.lmdbstore import LMDBStoreV3  # noqa: F401
from zarr._storage.lrustore import LRUStoreCacheV3  # noqa: F401
from zarr._storage.memorystore import KVStoreV3, MemoryStoreV3  # noqa: F401
from zarr._storage.mongodbstore import MongoDBStoreV3  # noqa: F401
from zarr._storage.redisstore import RedisStoreV3  # noqa: F401
from zarr._storage.sqlitestore import SQLiteStoreV3  # noqa: F401
from zarr._storage.store import BaseStore
from zarr._storage.store_v3 import data_root, meta_root, StoreV3  # noqa: F401
from zarr._storage.zipstore import ZipStoreV3  # noqa: F401


def _normalize_store_arg_v3(store: Any, storage_options=None, mode="r") -> BaseStore:
    # default to v2 store for backward compatibility
    zarr_version = getattr(store, '_store_version', 3)
    if zarr_version != 3:
        raise ValueError("store must be a version 3 store")
    if store is None:
        store = KVStoreV3(dict())
        # add default zarr.json metadata
        store['zarr.json'] = store._metadata_class.encode_hierarchy_metadata(None)
        return store
    if isinstance(store, os.PathLike):
        store = os.fspath(store)
    if isinstance(store, str):
        if "://" in store or "::" in store:
            store = FSStoreV3(store, mode=mode, **(storage_options or {}))
        elif storage_options:
            raise ValueError("storage_options passed with non-fsspec path")
        elif store.endswith('.zip'):
            store = ZipStoreV3(store, mode=mode)
        elif store.endswith('.n5'):
            raise NotImplementedError("N5Store not yet implemented for V3")
            # return N5StoreV3(store)
        else:
            store = DirectoryStoreV3(store)
        # add default zarr.json metadata
        store['zarr.json'] = store._metadata_class.encode_hierarchy_metadata(None)
        return store
    else:
        store = StoreV3._ensure_store(store)
        if 'zarr.json' not in store:
            # add default zarr.json metadata
            store['zarr.json'] = store._metadata_class.encode_hierarchy_metadata(None)
    return store
