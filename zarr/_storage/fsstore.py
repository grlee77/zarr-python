"""This module contains storage classes related to fsspec"""
import os

from zarr._storage.store import (_prefix_to_array_key, _prefix_to_group_key,
                                 _prog_number, array_meta_key, attrs_key, group_meta_key, Store)
from zarr._storage.store_v3 import _get_metadata_suffix, data_root, meta_root, StoreV3
from zarr.errors import FSPathExistNotDir, ReadOnlyError
from zarr.util import buffer_size, normalize_storage_path


class FSStore(Store):
    """Wraps an fsspec.FSMap to give access to arbitrary filesystems

    Requires that ``fsspec`` is installed, as well as any additional
    requirements for the protocol chosen.

    Parameters
    ----------
    url : str
        The destination to map. Should include protocol and path,
        like "s3://bucket/root"
    normalize_keys : bool
    key_separator : str
        public API for accessing dimension_separator. Never `None`
        See dimension_separator for more information.
    mode : str
        "w" for writable, "r" for read-only
    exceptions : list of Exception subclasses
        When accessing data, any of these exceptions will be treated
        as a missing key
    dimension_separator : {'.', '/'}, optional
        Separator placed between the dimensions of a chunk.
    storage_options : passed to the fsspec implementation
    """
    _array_meta_key = array_meta_key
    _group_meta_key = group_meta_key
    _attrs_key = attrs_key

    def __init__(self, url, normalize_keys=False, key_separator=None,
                 mode='w',
                 exceptions=(KeyError, PermissionError, IOError),
                 dimension_separator=None,
                 **storage_options):
        import fsspec
        self.normalize_keys = normalize_keys

        protocol, _ = fsspec.core.split_protocol(url)
        # set auto_mkdir to True for local file system
        if protocol in (None, "file") and not storage_options.get("auto_mkdir"):
            storage_options["auto_mkdir"] = True

        self.map = fsspec.get_mapper(url, **storage_options)
        self.fs = self.map.fs  # for direct operations
        self.path = self.fs._strip_protocol(url)
        self.mode = mode
        self.exceptions = exceptions
        # For backwards compatibility. Guaranteed to be non-None
        if key_separator is not None:
            dimension_separator = key_separator

        self.key_separator = dimension_separator
        self._default_key_separator()

        # Pass attributes to array creation
        self._dimension_separator = dimension_separator
        if self.fs.exists(self.path) and not self.fs.isdir(self.path):
            raise FSPathExistNotDir(url)

    def _default_key_separator(self):
        if self.key_separator is None:
            self.key_separator = "."

    def _normalize_key(self, key):
        key = normalize_storage_path(key).lstrip('/')
        if key:
            *bits, end = key.split('/')

            if end not in (self._array_meta_key, self._group_meta_key, self._attrs_key):
                end = end.replace('.', self.key_separator)
                key = '/'.join(bits + [end])

        return key.lower() if self.normalize_keys else key

    def getitems(self, keys, **kwargs):
        keys_transformed = [self._normalize_key(key) for key in keys]
        results = self.map.getitems(keys_transformed, on_error="omit")
        # The function calling this method may not recognize the transformed keys
        # So we send the values returned by self.map.getitems back into the original key space.
        return {keys[keys_transformed.index(rk)]: rv for rk, rv in results.items()}

    def __getitem__(self, key):
        key = self._normalize_key(key)
        try:
            return self.map[key]
        except self.exceptions as e:
            raise KeyError(key) from e

    def setitems(self, values):
        if self.mode == 'r':
            raise ReadOnlyError()
        values = {self._normalize_key(key): val for key, val in values.items()}
        self.map.setitems(values)

    def __setitem__(self, key, value):
        if self.mode == 'r':
            raise ReadOnlyError()
        key = self._normalize_key(key)
        path = self.dir_path(key)
        try:
            if self.fs.isdir(path):
                self.fs.rm(path, recursive=True)
            self.map[key] = value
            self.fs.invalidate_cache(self.fs._parent(path))
        except self.exceptions as e:
            raise KeyError(key) from e

    def __delitem__(self, key):
        if self.mode == 'r':
            raise ReadOnlyError()
        key = self._normalize_key(key)
        path = self.dir_path(key)
        if self.fs.isdir(path):
            self.fs.rm(path, recursive=True)
        else:
            del self.map[key]

    def delitems(self, keys):
        if self.mode == 'r':
            raise ReadOnlyError()
        # only remove the keys that exist in the store
        nkeys = [self._normalize_key(key) for key in keys if key in self]
        # rm errors if you pass an empty collection
        if len(nkeys) > 0:
            self.map.delitems(nkeys)

    def __contains__(self, key):
        key = self._normalize_key(key)
        return key in self.map

    def __eq__(self, other):
        return (type(self) == type(other) and self.map == other.map
                and self.mode == other.mode)

    def keys(self):
        return iter(self.map)

    def __iter__(self):
        return self.keys()

    def __len__(self):
        return len(list(self.keys()))

    def dir_path(self, path=None):
        store_path = normalize_storage_path(path)
        return self.map._key_to_str(store_path)

    def listdir(self, path=None):
        dir_path = self.dir_path(path)
        try:
            children = sorted(p.rstrip('/').rsplit('/', 1)[-1]
                              for p in self.fs.ls(dir_path, detail=False))
            if self.key_separator != "/":
                return children
            else:
                if self._array_meta_key in children:
                    # special handling of directories containing an array to map nested chunk
                    # keys back to standard chunk keys
                    new_children = []
                    root_path = self.dir_path(path)
                    for entry in children:
                        entry_path = os.path.join(root_path, entry)
                        if _prog_number.match(entry) and self.fs.isdir(entry_path):
                            for file_name in self.fs.find(entry_path):
                                file_path = os.path.join(dir_path, file_name)
                                rel_path = file_path.split(root_path)[1]
                                rel_path = rel_path.lstrip('/')
                                new_children.append(rel_path.replace('/', '.'))
                        else:
                            new_children.append(entry)
                    return sorted(new_children)
                else:
                    return children
        except IOError:
            return []

    def rmdir(self, path=None):
        if self.mode == 'r':
            raise ReadOnlyError()
        store_path = self.dir_path(path)
        if self.fs.isdir(store_path):
            self.fs.rm(store_path, recursive=True)

    def getsize(self, path=None):
        store_path = self.dir_path(path)
        return self.fs.du(store_path, True, True)

    def clear(self):
        if self.mode == 'r':
            raise ReadOnlyError()
        self.map.clear()


def _get_files_and_dirs_from_path(store, path):
    path = normalize_storage_path(path)

    files = []
    # add array metadata file if present
    array_key = _prefix_to_array_key(store, path)
    if array_key in store:
        files.append(os.path.join(store.path, array_key))

    # add group metadata file if present
    group_key = _prefix_to_group_key(store, path)
    if group_key in store:
        files.append(os.path.join(store.path, group_key))

    dirs = []
    # add array and group folders if present
    for d in [data_root + path, meta_root + path]:
        dir_path = os.path.join(store.path, d)
        if os.path.exists(dir_path):
            dirs.append(dir_path)
    return files, dirs


class FSStoreV3(FSStore, StoreV3):

    # FSStoreV3 doesn't use this (FSStore uses it within _normalize_key)
    _META_KEYS = ()

    def __setitem__(self, key, value):
        self._validate_key(key)
        super().__setitem__(key, value)

    def _default_key_separator(self):
        if self.key_separator is None:
            self.key_separator = "/"

    def list(self):
        return list(self.keys())

    def _normalize_key(self, key):
        key = normalize_storage_path(key).lstrip('/')
        return key.lower() if self.normalize_keys else key

    def getsize(self, path=None):
        size = 0
        if path is None or path == '':
            # size of both the data and meta subdirs
            dirs = []
            for d in ['data/root', 'meta/root']:
                dir_path = os.path.join(self.path, d)
                if os.path.exists(dir_path):
                    dirs.append(dir_path)
        elif path in self:
            # access individual element by full path
            return buffer_size(self[path])
        else:
            files, dirs = _get_files_and_dirs_from_path(self, path)
            for file in files:
                size += os.path.getsize(file)
        for d in dirs:
            size += self.fs.du(d, total=True, maxdepth=None)
        return size

    def setitems(self, values):
        if self.mode == 'r':
            raise ReadOnlyError()
        values = {self._normalize_key(key): val for key, val in values.items()}

        # initialize the /data/root/... folder corresponding to the array!
        # Note: zarr.tests.test_core_v3.TestArrayWithFSStoreV3PartialRead fails
        # without this explicit creation of directories
        subdirectories = set([os.path.dirname(v) for v in values.keys()])
        for subdirectory in subdirectories:
            data_dir = os.path.join(self.path, subdirectory)
            if not self.fs.exists(data_dir):
                self.fs.mkdir(data_dir)

        self.map.setitems(values)

    def rmdir(self, path=None):
        if self.mode == 'r':
            raise ReadOnlyError()
        if path:
            for base in [meta_root, data_root]:
                store_path = self.dir_path(base + path)
                if self.fs.isdir(store_path):
                    self.fs.rm(store_path, recursive=True)

            # remove any associated metadata files
            sfx = _get_metadata_suffix(self)
            meta_dir = (meta_root + path).rstrip('/')
            array_meta_file = meta_dir + '.array' + sfx
            self.pop(array_meta_file, None)
            group_meta_file = meta_dir + '.group' + sfx
            self.pop(group_meta_file, None)
        else:
            store_path = self.dir_path(path)
            if self.fs.isdir(store_path):
                self.fs.rm(store_path, recursive=True)


FSStoreV3.__doc__ = FSStore.__doc__
