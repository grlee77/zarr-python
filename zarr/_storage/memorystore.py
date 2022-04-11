"""This module contains storage classes holding all data in main memory."""
import warnings
from threading import Lock
from typing import Dict, List

from numcodecs.compat import ensure_bytes

from zarr._storage.store import _getsize, Path, Store
from zarr._storage.store_v3 import (_get_metadata_suffix, _rename_metadata_v3, data_root,
                                    meta_root, StoreV3)
from zarr.util import buffer_size, normalize_storage_path


def _dict_store_keys(d: Dict, prefix="", cls=dict):
    for k in d.keys():
        v = d[k]
        if isinstance(v, cls):
            for sk in _dict_store_keys(v, prefix + k + '/', cls):
                yield sk
        else:
            yield prefix + k


class MemoryStore(Store):
    """Store class that uses a hierarchy of :class:`KVStore` objects, thus all data
    will be held in main memory.

    Examples
    --------
    This is the default class used when creating a group. E.g.::

        >>> import zarr
        >>> g = zarr.group()
        >>> type(g.store)
        <class 'zarr.storage.MemoryStore'>

    Note that the default class when creating an array is the built-in
    :class:`KVStore` class, i.e.::

        >>> z = zarr.zeros(100)
        >>> type(z.store)
        <class 'zarr.storage.KVStore'>

    Notes
    -----
    Safe to write in multiple threads.

    """

    def __init__(self, root=None, cls=dict, dimension_separator=None):
        if root is None:
            self.root = cls()
        else:
            self.root = root
        self.cls = cls
        self.write_mutex = Lock()
        self._dimension_separator = dimension_separator

    def __getstate__(self):
        return self.root, self.cls

    def __setstate__(self, state):
        root, cls = state
        self.__init__(root=root, cls=cls)

    def _get_parent(self, item: str):
        parent = self.root
        # split the item
        segments = item.split('/')
        # find the parent container
        for k in segments[:-1]:
            parent = parent[k]
            if not isinstance(parent, self.cls):
                raise KeyError(item)
        return parent, segments[-1]

    def _require_parent(self, item):
        parent = self.root
        # split the item
        segments = item.split('/')
        # require the parent container
        for k in segments[:-1]:
            try:
                parent = parent[k]
            except KeyError:
                parent[k] = self.cls()
                parent = parent[k]
            else:
                if not isinstance(parent, self.cls):
                    raise KeyError(item)
        return parent, segments[-1]

    def __getitem__(self, item: str):
        parent, key = self._get_parent(item)
        try:
            value = parent[key]
        except KeyError:
            raise KeyError(item)
        else:
            if isinstance(value, self.cls):
                raise KeyError(item)
            else:
                return value

    def __setitem__(self, item: str, value):
        with self.write_mutex:
            parent, key = self._require_parent(item)
            value = ensure_bytes(value)
            parent[key] = value

    def __delitem__(self, item: str):
        with self.write_mutex:
            parent, key = self._get_parent(item)
            try:
                del parent[key]
            except KeyError:
                raise KeyError(item)

    def __contains__(self, item: str):  # type: ignore[override]
        try:
            parent, key = self._get_parent(item)
            value = parent[key]
        except KeyError:
            return False
        else:
            return not isinstance(value, self.cls)

    def __eq__(self, other):
        return (
            isinstance(other, MemoryStore) and
            self.root == other.root and
            self.cls == other.cls
        )

    def keys(self):
        for k in _dict_store_keys(self.root, cls=self.cls):
            yield k

    def __iter__(self):
        return self.keys()

    def __len__(self) -> int:
        return sum(1 for _ in self.keys())

    def listdir(self, path: Path = None) -> List[str]:
        path = normalize_storage_path(path)
        if path:
            try:
                parent, key = self._get_parent(path)
                value = parent[key]
            except KeyError:
                return []
        else:
            value = self.root
        if isinstance(value, self.cls):
            return sorted(value.keys())
        else:
            return []

    def rename(self, src_path: Path, dst_path: Path):
        src_path = normalize_storage_path(src_path)
        dst_path = normalize_storage_path(dst_path)

        src_parent, src_key = self._get_parent(src_path)
        dst_parent, dst_key = self._require_parent(dst_path)

        dst_parent[dst_key] = src_parent.pop(src_key)

    def rmdir(self, path: Path = None):
        path = normalize_storage_path(path)
        if path:
            try:
                parent, key = self._get_parent(path)
                value = parent[key]
            except KeyError:
                return
            else:
                if isinstance(value, self.cls):
                    del parent[key]
        else:
            # clear out root
            self.root = self.cls()

    def getsize(self, path: Path = None):
        path = normalize_storage_path(path)

        # obtain value to return size of
        value = None
        if path:
            try:
                parent, key = self._get_parent(path)
                value = parent[key]
            except KeyError:
                pass
        else:
            value = self.root

        # obtain size of value
        if value is None:
            return 0

        elif isinstance(value, self.cls):
            # total size for directory
            size = 0
            for v in value.values():
                if not isinstance(v, self.cls):
                    size += buffer_size(v)
            return size

        else:
            return buffer_size(value)

    def clear(self):
        with self.write_mutex:
            self.root.clear()


class DictStore(MemoryStore):

    def __init__(self, *args, **kwargs):
        warnings.warn("DictStore has been renamed to MemoryStore in 2.4.0 and "
                      "will be removed in the future. Please use MemoryStore.",
                      DeprecationWarning,
                      stacklevel=2)
        super().__init__(*args, **kwargs)


class MemoryStoreV3(MemoryStore, StoreV3):

    def __init__(self, root=None, cls=dict, dimension_separator=None):
        if root is None:
            self.root = cls()
        else:
            self.root = root
        self.cls = cls
        self.write_mutex = Lock()
        self._dimension_separator = dimension_separator  # TODO: modify for v3?

    def __eq__(self, other):
        return (
            isinstance(other, MemoryStoreV3) and
            self.root == other.root and
            self.cls == other.cls
        )

    def __setitem__(self, key, value):
        self._validate_key(key)
        super().__setitem__(key, value)

    def list(self):
        return list(self.keys())

    def getsize(self, path: Path = None):
        return _getsize(self, path)

    def rename(self, src_path: Path, dst_path: Path):
        src_path = normalize_storage_path(src_path)
        dst_path = normalize_storage_path(dst_path)

        any_renamed = False
        for base in [meta_root, data_root]:
            if self.list_prefix(base + src_path):
                src_parent, src_key = self._get_parent(base + src_path)
                dst_parent, dst_key = self._require_parent(base + dst_path)

                if src_key in src_parent:
                    dst_parent[dst_key] = src_parent.pop(src_key)

                if base == meta_root:
                    # check for and move corresponding metadata
                    sfx = _get_metadata_suffix(self)
                    src_meta = src_key + '.array' + sfx
                    if src_meta in src_parent:
                        dst_meta = dst_key + '.array' + sfx
                        dst_parent[dst_meta] = src_parent.pop(src_meta)
                    src_meta = src_key + '.group' + sfx
                    if src_meta in src_parent:
                        dst_meta = dst_key + '.group' + sfx
                        dst_parent[dst_meta] = src_parent.pop(src_meta)
                any_renamed = True
        any_renamed = _rename_metadata_v3(self, src_path, dst_path) or any_renamed
        if not any_renamed:
            raise ValueError(f"no item {src_path} found to rename")

    def rmdir(self, path: Path = None):
        path = normalize_storage_path(path)
        if path:
            for base in [meta_root, data_root]:
                try:
                    parent, key = self._get_parent(base + path)
                    value = parent[key]
                except KeyError:
                    continue
                else:
                    if isinstance(value, self.cls):
                        del parent[key]

            # remove any associated metadata files
            sfx = _get_metadata_suffix(self)
            meta_dir = (meta_root + path).rstrip('/')
            array_meta_file = meta_dir + '.array' + sfx
            self.pop(array_meta_file, None)
            group_meta_file = meta_dir + '.group' + sfx
            self.pop(group_meta_file, None)
        else:
            # clear out root
            self.root = self.cls()


MemoryStoreV3.__doc__ = MemoryStore.__doc__
