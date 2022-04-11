"""This module contains storage classes for use with Zarr arrays and groups.

Note that any object implementing the :class:`MutableMapping` interface from the
:mod:`collections` module in the Python standard library can be used as a Zarr
array store, as long as it accepts string (str) keys and bytes values.

In addition to the :class:`MutableMapping` interface, store classes may also implement
optional methods `listdir` (list members of a "directory") and `rmdir` (remove all
members of a "directory"). These methods should be implemented if the store class is
aware of the hierarchical organisation of resources within the store and can provide
efficient implementations. If these methods are not available, Zarr will fall back to
slower implementations that work via the :class:`MutableMapping` interface. Store
classes may also optionally implement a `rename` method (rename all members under a given
path) and a `getsize` method (return the size in bytes of a given value).

"""
import glob
import os
import shutil
import warnings
from collections.abc import MutableMapping
from typing import Optional, Union, Tuple, Any

from numcodecs.abc import Codec
from numcodecs.registry import codec_registry

from zarr.errors import BadCompressorError, ContainsArrayError, ContainsGroupError
from zarr.meta import encode_array_metadata, encode_group_metadata
from zarr.util import (normalize_chunks, normalize_dimension_separator,
                       normalize_dtype, normalize_fill_value, normalize_order,
                       normalize_shape, normalize_storage_path)

from zarr._storage.absstore import ABSStore  # noqa: F401
from zarr._storage.consolidatedstore import ConsolidatedMetadataStore  # noqa: F401
from zarr._storage.directorystore import (atexit_rmtree, DirectoryStore,  # noqa: F401
                                          NestedDirectoryStore, TempStore)
from zarr._storage.dbmstore import DBMStore  # noqa: F401
from zarr._storage.fsstore import FSStore  # noqa: F401
from zarr._storage.kvstore import KVStore  # noqa: F401
from zarr._storage.lmdbstore import LMDBStore  # noqa: F401
from zarr._storage.lrustore import LRUStoreCache  # noqa: F401
from zarr._storage.memorystore import DictStore, MemoryStore  # noqa: F401
from zarr._storage.mongodbstore import MongoDBStore  # noqa: F401
from zarr._storage.redisstore import RedisStore  # noqa: F401
from zarr._storage.sqlitestore import SQLiteStore  # noqa: F401
from zarr._storage.zipstore import ZipStore  # noqa: F401
from zarr._storage.store import (_rename_from_keys,  # noqa: F401
                                 _rmdir_from_keys,
                                 _path_to_prefix,
                                 _prefix_to_array_key,
                                 _prefix_to_group_key,
                                 array_meta_key,
                                 attrs_key,
                                 getsize,
                                 group_meta_key,
                                 listdir,
                                 DEFAULT_ZARR_VERSION,
                                 BaseStore,
                                 Store)
from zarr._storage.store_v3 import (_get_metadata_suffix, _get_hierarchy_metadata,
                                    _rmdir_from_keys_v3, data_root, meta_root,
                                    v3_api_available)


try:
    # noinspection PyUnresolvedReferences
    from zarr.codecs import Blosc
    default_compressor = Blosc()
except ImportError:  # pragma: no cover
    from zarr.codecs import Zlib
    default_compressor = Zlib()


Path = Union[str, bytes, None]
# allow MutableMapping for backwards compatibility
StoreLike = Union[BaseStore, MutableMapping]


def contains_array(store: StoreLike, path: Path = None) -> bool:
    """Return True if the store contains an array at the given logical path."""
    path = normalize_storage_path(path)
    prefix = _path_to_prefix(path)
    key = _prefix_to_array_key(store, prefix)
    return key in store


def contains_group(store: StoreLike, path: Path = None, explicit_only=True) -> bool:
    """Return True if the store contains a group at the given logical path."""
    path = normalize_storage_path(path)
    prefix = _path_to_prefix(path)
    key = _prefix_to_group_key(store, prefix)
    store_version = getattr(store, '_store_version', 2)
    if store_version == 2 or explicit_only:
        return key in store
    else:
        if key in store:
            return True
        # for v3, need to also handle implicit groups

        sfx = _get_metadata_suffix(store)  # type: ignore
        implicit_prefix = key.replace('.group' + sfx, '')
        if not implicit_prefix.endswith('/'):
            implicit_prefix += '/'
        if store.list_prefix(implicit_prefix):  # type: ignore
            return True
        return False


def _normalize_store_arg_v2(store: Any, storage_options=None, mode="r") -> BaseStore:
    # default to v2 store for backward compatibility
    zarr_version = getattr(store, '_store_version', 2)
    if zarr_version != 2:
        raise ValueError("store must be a version 2 store")
    if store is None:
        store = KVStore(dict())
        return store
    if isinstance(store, os.PathLike):
        store = os.fspath(store)
    if isinstance(store, str):
        if "://" in store or "::" in store:
            return FSStore(store, mode=mode, **(storage_options or {}))
        elif storage_options:
            raise ValueError("storage_options passed with non-fsspec path")
        if store.endswith('.zip'):
            return ZipStore(store, mode=mode)
        elif store.endswith('.n5'):
            from zarr.n5 import N5Store
            return N5Store(store)
        else:
            return DirectoryStore(store)
    else:
        store = Store._ensure_store(store)
    return store


def normalize_store_arg(store: Any, storage_options=None, mode="r", *,
                        zarr_version=None) -> BaseStore:
    if zarr_version is None:
        # default to v2 store for backward compatibility
        zarr_version = getattr(store, "_store_version", DEFAULT_ZARR_VERSION)
    elif zarr_version not in [2, 3]:
        raise ValueError("zarr_version must be either 2 or 3")
    if zarr_version == 2:
        normalize_store = _normalize_store_arg_v2
    elif zarr_version == 3:
        from zarr.storage_v3 import _normalize_store_arg_v3 as normalize_store
    return normalize_store(store, storage_options, mode)


def rmdir(store: StoreLike, path: Path = None):
    """Remove all items under the given path. If `store` provides a `rmdir` method,
    this will be called, otherwise will fall back to implementation via the
    `Store` interface."""
    path = normalize_storage_path(path)
    store_version = getattr(store, '_store_version', 2)
    if hasattr(store, "rmdir") and store.is_erasable():  # type: ignore
        # pass through
        store.rmdir(path)  # type: ignore
    else:
        # slow version, delete one key at a time
        if store_version == 2:
            _rmdir_from_keys(store, path)
        else:
            _rmdir_from_keys_v3(store, path)  # type: ignore


def rename(store: Store, src_path: Path, dst_path: Path):
    """Rename all items under the given path. If `store` provides a `rename` method,
    this will be called, otherwise will fall back to implementation via the
    `Store` interface."""
    src_path = normalize_storage_path(src_path)
    dst_path = normalize_storage_path(dst_path)
    if hasattr(store, 'rename'):
        # pass through
        store.rename(src_path, dst_path)
    else:
        # slow version, delete one key at a time
        _rename_from_keys(store, src_path, dst_path)


def _require_parent_group(
    path: Optional[str],
    store: StoreLike,
    chunk_store: Optional[StoreLike],
    overwrite: bool,
):
    # assume path is normalized
    if path:
        segments = path.split('/')
        for i in range(len(segments)):
            p = '/'.join(segments[:i])
            if contains_array(store, p):
                _init_group_metadata(store, path=p, chunk_store=chunk_store,
                                     overwrite=overwrite)
            elif not contains_group(store, p):
                _init_group_metadata(store, path=p, chunk_store=chunk_store)


def init_array(
    store: StoreLike,
    shape: Tuple[int, ...],
    chunks: Union[bool, int, Tuple[int, ...]] = True,
    dtype=None,
    compressor="default",
    fill_value=None,
    order: str = "C",
    overwrite: bool = False,
    path: Optional[Path] = None,
    chunk_store: Optional[StoreLike] = None,
    filters=None,
    object_codec=None,
    dimension_separator=None,
):
    """Initialize an array store with the given configuration. Note that this is a low-level
    function and there should be no need to call this directly from user code.

    Parameters
    ----------
    store : Store
        A mapping that supports string keys and bytes-like values.
    shape : int or tuple of ints
        Array shape.
    chunks : bool, int or tuple of ints, optional
        Chunk shape. If True, will be guessed from `shape` and `dtype`. If
        False, will be set to `shape`, i.e., single chunk for the whole array.
    dtype : string or dtype, optional
        NumPy dtype.
    compressor : Codec, optional
        Primary compressor.
    fill_value : object
        Default value to use for uninitialized portions of the array.
    order : {'C', 'F'}, optional
        Memory layout to be used within each chunk.
    overwrite : bool, optional
        If True, erase all data in `store` prior to initialisation.
    path : string, bytes, optional
        Path under which array is stored.
    chunk_store : Store, optional
        Separate storage for chunks. If not provided, `store` will be used
        for storage of both chunks and metadata.
    filters : sequence, optional
        Sequence of filters to use to encode chunk data prior to compression.
    object_codec : Codec, optional
        A codec to encode object arrays, only needed if dtype=object.
    dimension_separator : {'.', '/'}, optional
        Separator placed between the dimensions of a chunk.

    Examples
    --------
    Initialize an array store::

        >>> from zarr.storage import init_array, KVStore
        >>> store = KVStore(dict())
        >>> init_array(store, shape=(10000, 10000), chunks=(1000, 1000))
        >>> sorted(store.keys())
        ['.zarray']

    Array metadata is stored as JSON::

        >>> print(store['.zarray'].decode())
        {
            "chunks": [
                1000,
                1000
            ],
            "compressor": {
                "blocksize": 0,
                "clevel": 5,
                "cname": "lz4",
                "id": "blosc",
                "shuffle": 1
            },
            "dtype": "<f8",
            "fill_value": null,
            "filters": null,
            "order": "C",
            "shape": [
                10000,
                10000
            ],
            "zarr_format": 2
        }

    Initialize an array using a storage path::

        >>> store = KVStore(dict())
        >>> init_array(store, shape=100000000, chunks=1000000, dtype='i1', path='foo')
        >>> sorted(store.keys())
        ['.zgroup', 'foo/.zarray']
        >>> print(store['foo/.zarray'].decode())
        {
            "chunks": [
                1000000
            ],
            "compressor": {
                "blocksize": 0,
                "clevel": 5,
                "cname": "lz4",
                "id": "blosc",
                "shuffle": 1
            },
            "dtype": "|i1",
            "fill_value": null,
            "filters": null,
            "order": "C",
            "shape": [
                100000000
            ],
            "zarr_format": 2
        }

    Notes
    -----
    The initialisation process involves normalising all array metadata, encoding
    as JSON and storing under the '.zarray' key.

    """

    # normalize path
    path = normalize_storage_path(path)

    # ensure parent group initialized
    store_version = getattr(store, "_store_version", 2)
    if store_version < 3:
        _require_parent_group(path, store=store, chunk_store=chunk_store,
                              overwrite=overwrite)

    if store_version == 3 and 'zarr.json' not in store:
        # initialize with default zarr.json entry level metadata
        store['zarr.json'] = store._metadata_class.encode_hierarchy_metadata(None)  # type: ignore

    if not compressor:
        # compatibility with legacy tests using compressor=[]
        compressor = None
    _init_array_metadata(store, shape=shape, chunks=chunks, dtype=dtype,
                         compressor=compressor, fill_value=fill_value,
                         order=order, overwrite=overwrite, path=path,
                         chunk_store=chunk_store, filters=filters,
                         object_codec=object_codec,
                         dimension_separator=dimension_separator)


def _init_array_metadata(
    store: StoreLike,
    shape,
    chunks=None,
    dtype=None,
    compressor="default",
    fill_value=None,
    order="C",
    overwrite=False,
    path: Optional[str] = None,
    chunk_store: Optional[StoreLike] = None,
    filters=None,
    object_codec=None,
    dimension_separator=None,
):

    store_version = getattr(store, '_store_version', 2)

    path = normalize_storage_path(path)

    # guard conditions
    if overwrite:
        if store_version == 2:
            # attempt to delete any pre-existing array in store
            rmdir(store, path)
            if chunk_store is not None:
                rmdir(chunk_store, path)
        else:
            group_meta_key_v3 = _prefix_to_group_key(store, _path_to_prefix(path))
            array_meta_key_v3 = _prefix_to_array_key(store, _path_to_prefix(path))
            data_prefix = data_root + _path_to_prefix(path)

            # attempt to delete any pre-existing array in store
            if array_meta_key_v3 in store:
                store.erase(array_meta_key_v3)  # type: ignore
            if group_meta_key_v3 in store:
                store.erase(group_meta_key_v3)  # type: ignore
            store.erase_prefix(data_prefix)  # type: ignore
            if chunk_store is not None:
                chunk_store.erase_prefix(data_prefix)  # type: ignore

            if '/' in path:
                # path is a subfolder of an existing array, remove that array
                parent_path = '/'.join(path.split('/')[:-1])
                sfx = _get_metadata_suffix(store)  # type: ignore
                array_key = meta_root + parent_path + '.array' + sfx
                if array_key in store:
                    store.erase(array_key)  # type: ignore

    if not overwrite:
        if contains_array(store, path):
            raise ContainsArrayError(path)
        elif contains_group(store, path, explicit_only=False):
            raise ContainsGroupError(path)
        elif store_version == 3:
            if '/' in path:
                # cannot create an array within an existing array path
                parent_path = '/'.join(path.split('/')[:-1])
                if contains_array(store, parent_path):
                    raise ContainsArrayError(path)

    # normalize metadata
    dtype, object_codec = normalize_dtype(dtype, object_codec)
    shape = normalize_shape(shape) + dtype.shape
    dtype = dtype.base
    chunks = normalize_chunks(chunks, shape, dtype.itemsize)
    order = normalize_order(order)
    fill_value = normalize_fill_value(fill_value, dtype)

    # optional array metadata
    if dimension_separator is None and store_version == 2:
        dimension_separator = getattr(store, "_dimension_separator", None)
    dimension_separator = normalize_dimension_separator(dimension_separator)

    # compressor prep
    if shape == ():
        # no point in compressing a 0-dimensional array, only a single value
        compressor = None
    elif compressor == 'none':
        # compatibility
        compressor = None
    elif compressor == 'default':
        compressor = default_compressor

    # obtain compressor config
    compressor_config = None
    if compressor:
        if store_version == 2:
            try:
                compressor_config = compressor.get_config()
            except AttributeError as e:
                raise BadCompressorError(compressor) from e
        elif not isinstance(compressor, Codec):
            raise ValueError("expected a numcodecs Codec for compressor")
            # TODO: alternatively, could autoconvert str to a Codec
            #       e.g. 'zlib' -> numcodec.Zlib object
            # compressor = numcodecs.get_codec({'id': compressor})

    # obtain filters config
    if filters:
        # TODO: filters was removed from the metadata in v3
        #       raise error here if store_version > 2?
        filters_config = [f.get_config() for f in filters]
    else:
        filters_config = []

    # deal with object encoding
    if dtype.hasobject:
        if object_codec is None:
            if not filters:
                # there are no filters so we can be sure there is no object codec
                raise ValueError('missing object_codec for object array')
            else:
                # one of the filters may be an object codec, issue a warning rather
                # than raise an error to maintain backwards-compatibility
                warnings.warn('missing object_codec for object array; this will raise a '
                              'ValueError in version 3.0', FutureWarning)
        else:
            filters_config.insert(0, object_codec.get_config())
    elif object_codec is not None:
        warnings.warn('an object_codec is only needed for object arrays')

    # use null to indicate no filters
    if not filters_config:
        filters_config = None  # type: ignore

    # initialize metadata
    # TODO: don't store redundant dimension_separator for v3?
    _compressor = compressor_config if store_version == 2 else compressor
    meta = dict(shape=shape, compressor=_compressor,
                fill_value=fill_value,
                dimension_separator=dimension_separator)
    if store_version < 3:
        meta.update(dict(chunks=chunks, dtype=dtype, order=order,
                         filters=filters_config))
    else:
        if dimension_separator is None:
            dimension_separator = "/"
        if filters_config:
            attributes = {'filters': filters_config}
        else:
            attributes = {}
        meta.update(
            dict(chunk_grid=dict(type="regular",
                                 chunk_shape=chunks,
                                 separator=dimension_separator),
                 chunk_memory_layout=order,
                 data_type=dtype,
                 attributes=attributes)
        )

    key = _prefix_to_array_key(store, _path_to_prefix(path))
    if hasattr(store, '_metadata_class'):
        store[key] = store._metadata_class.encode_array_metadata(meta)  # type: ignore
    else:
        store[key] = encode_array_metadata(meta)


# backwards compatibility
init_store = init_array


def init_group(
    store: StoreLike,
    overwrite: bool = False,
    path: Path = None,
    chunk_store: StoreLike = None,
):
    """Initialize a group store. Note that this is a low-level function and there should be no
    need to call this directly from user code.

    Parameters
    ----------
    store : Store
        A mapping that supports string keys and byte sequence values.
    overwrite : bool, optional
        If True, erase all data in `store` prior to initialisation.
    path : string, optional
        Path under which array is stored.
    chunk_store : Store, optional
        Separate storage for chunks. If not provided, `store` will be used
        for storage of both chunks and metadata.

    """

    # normalize path
    path = normalize_storage_path(path)

    store_version = getattr(store, '_store_version', 2)
    if store_version < 3:
        # ensure parent group initialized
        _require_parent_group(path, store=store, chunk_store=chunk_store,
                              overwrite=overwrite)

    if store_version == 3 and 'zarr.json' not in store:
        # initialize with default zarr.json entry level metadata
        store['zarr.json'] = store._metadata_class.encode_hierarchy_metadata(None)  # type: ignore

    # initialise metadata
    _init_group_metadata(store=store, overwrite=overwrite, path=path,
                         chunk_store=chunk_store)

    if store_version == 3:
        # TODO: Should initializing a v3 group also create a corresponding
        #       empty folder under data/root/? I think probably not until there
        #       is actual data written there.
        pass


def _init_group_metadata(
    store: StoreLike,
    overwrite: Optional[bool] = False,
    path: Optional[str] = None,
    chunk_store: StoreLike = None,
):

    store_version = getattr(store, '_store_version', 2)
    path = normalize_storage_path(path)

    # guard conditions
    if overwrite:
        if store_version == 2:
            # attempt to delete any pre-existing items in store
            rmdir(store, path)
            if chunk_store is not None:
                rmdir(chunk_store, path)
        else:
            group_meta_key_v3 = _prefix_to_group_key(store, _path_to_prefix(path))
            array_meta_key_v3 = _prefix_to_array_key(store, _path_to_prefix(path))
            data_prefix = data_root + _path_to_prefix(path)
            meta_prefix = meta_root + _path_to_prefix(path)

            # attempt to delete any pre-existing array in store
            if array_meta_key_v3 in store:
                store.erase(array_meta_key_v3)  # type: ignore
            if group_meta_key_v3 in store:
                store.erase(group_meta_key_v3)  # type: ignore
            store.erase_prefix(data_prefix)  # type: ignore
            store.erase_prefix(meta_prefix)  # type: ignore
            if chunk_store is not None:
                chunk_store.erase_prefix(data_prefix)  # type: ignore

    if not overwrite:
        if contains_array(store, path):
            raise ContainsArrayError(path)
        elif contains_group(store, path):
            raise ContainsGroupError(path)
        elif store_version == 3 and '/' in path:
            # cannot create a group overlapping with an existing array name
            parent_path = '/'.join(path.split('/')[:-1])
            if contains_array(store, parent_path):
                raise ContainsArrayError(path)

    # initialize metadata
    # N.B., currently no metadata properties are needed, however there may
    # be in future
    if store_version == 3:
        meta = {'attributes': {}}  # type: ignore
    else:
        meta = {}  # type: ignore
    key = _prefix_to_group_key(store, _path_to_prefix(path))
    if hasattr(store, '_metadata_class'):
        store[key] = store._metadata_class.encode_group_metadata(meta)  # type: ignore
    else:
        store[key] = encode_group_metadata(meta)


# noinspection PyShadowingNames
def atexit_rmglob(path,
                  glob=glob.glob,
                  isdir=os.path.isdir,
                  isfile=os.path.isfile,
                  remove=os.remove,
                  rmtree=shutil.rmtree):  # pragma: no cover
    """Ensure removal of multiple files at interpreter exit."""
    for p in glob(path):
        if isfile(p):
            remove(p)
        elif isdir(p):
            rmtree(p)


def migrate_1to2(store):
    """Migrate array metadata in `store` from Zarr format version 1 to
    version 2.

    Parameters
    ----------
    store : Store
        Store to be migrated.

    Notes
    -----
    Version 1 did not support hierarchies, so this migration function will
    look for a single array in `store` and migrate the array metadata to
    version 2.

    """

    # migrate metadata
    from zarr import meta_v1
    meta = meta_v1.decode_metadata(store['meta'])
    del store['meta']

    # add empty filters
    meta['filters'] = None

    # migration compression metadata
    compression = meta['compression']
    if compression is None or compression == 'none':
        compressor_config = None
    else:
        compression_opts = meta['compression_opts']
        codec_cls = codec_registry[compression]
        if isinstance(compression_opts, dict):
            compressor = codec_cls(**compression_opts)
        else:
            compressor = codec_cls(compression_opts)
        compressor_config = compressor.get_config()
    meta['compressor'] = compressor_config
    del meta['compression']
    del meta['compression_opts']

    # store migrated metadata
    if hasattr(store, '_metadata_class'):
        store[array_meta_key] = store._metadata_class.encode_array_metadata(meta)
    else:
        store[array_meta_key] = encode_array_metadata(meta)

    # migrate user attributes
    store[attrs_key] = store['attrs']
    del store['attrs']
