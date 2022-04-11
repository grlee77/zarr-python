"""This module contains storage classes related to a standard filesytem."""
import atexit
import errno
import os
import shutil
import tempfile
import uuid
from os import scandir

from numcodecs.compat import ensure_contiguous_ndarray

from zarr._storage.store import (_getsize, _prog_number, array_meta_key,
                                 Path, Store)
from zarr._storage.store_v3 import _get_metadata_suffix, data_root, meta_root, StoreV3
from zarr.errors import FSPathExistNotDir
from zarr.util import normalize_storage_path, retry_call


def atexit_rmtree(path,
                  isdir=os.path.isdir,
                  rmtree=shutil.rmtree):  # pragma: no cover
    """Ensure directory removal at interpreter exit."""
    if isdir(path):
        rmtree(path)


class DirectoryStore(Store):
    """Storage class using directories and files on a standard file system.

    Parameters
    ----------
    path : string
        Location of directory to use as the root of the storage hierarchy.
    normalize_keys : bool, optional
        If True, all store keys will be normalized to use lower case characters
        (e.g. 'foo' and 'FOO' will be treated as equivalent). This can be
        useful to avoid potential discrepancies between case-sensitive and
        case-insensitive file system. Default value is False.
    dimension_separator : {'.', '/'}, optional
        Separator placed between the dimensions of a chunk.

    Examples
    --------
    Store a single array::

        >>> import zarr
        >>> store = zarr.DirectoryStore('data/array.zarr')
        >>> z = zarr.zeros((10, 10), chunks=(5, 5), store=store, overwrite=True)
        >>> z[...] = 42

    Each chunk of the array is stored as a separate file on the file system,
    i.e.::

        >>> import os
        >>> sorted(os.listdir('data/array.zarr'))
        ['.zarray', '0.0', '0.1', '1.0', '1.1']

    Store a group::

        >>> store = zarr.DirectoryStore('data/group.zarr')
        >>> root = zarr.group(store=store, overwrite=True)
        >>> foo = root.create_group('foo')
        >>> bar = foo.zeros('bar', shape=(10, 10), chunks=(5, 5))
        >>> bar[...] = 42

    When storing a group, levels in the group hierarchy will correspond to
    directories on the file system, i.e.::

        >>> sorted(os.listdir('data/group.zarr'))
        ['.zgroup', 'foo']
        >>> sorted(os.listdir('data/group.zarr/foo'))
        ['.zgroup', 'bar']
        >>> sorted(os.listdir('data/group.zarr/foo/bar'))
        ['.zarray', '0.0', '0.1', '1.0', '1.1']

    Notes
    -----
    Atomic writes are used, which means that data are first written to a
    temporary file, then moved into place when the write is successfully
    completed. Files are only held open while they are being read or written and are
    closed immediately afterwards, so there is no need to manually close any files.

    Safe to write in multiple threads or processes.

    """

    def __init__(self, path, normalize_keys=False, dimension_separator=None):

        # guard conditions
        path = os.path.abspath(path)
        if os.path.exists(path) and not os.path.isdir(path):
            raise FSPathExistNotDir(path)

        self.path = path
        self.normalize_keys = normalize_keys
        self._dimension_separator = dimension_separator

    def _normalize_key(self, key):
        return key.lower() if self.normalize_keys else key

    @staticmethod
    def _fromfile(fn):
        """ Read data from a file

        Parameters
        ----------
        fn : str
            Filepath to open and read from.

        Notes
        -----
        Subclasses should overload this method to specify any custom
        file reading logic.
        """
        with open(fn, 'rb') as f:
            return f.read()

    @staticmethod
    def _tofile(a, fn):
        """ Write data to a file

        Parameters
        ----------
        a : array-like
            Data to write into the file.
        fn : str
            Filepath to open and write to.

        Notes
        -----
        Subclasses should overload this method to specify any custom
        file writing logic.
        """
        with open(fn, mode='wb') as f:
            f.write(a)

    def __getitem__(self, key):
        key = self._normalize_key(key)
        filepath = os.path.join(self.path, key)
        if os.path.isfile(filepath):
            return self._fromfile(filepath)
        else:
            raise KeyError(key)

    def __setitem__(self, key, value):
        key = self._normalize_key(key)

        # coerce to flat, contiguous array (ideally without copying)
        value = ensure_contiguous_ndarray(value)

        # destination path for key
        file_path = os.path.join(self.path, key)

        # ensure there is no directory in the way
        if os.path.isdir(file_path):
            shutil.rmtree(file_path)

        # ensure containing directory exists
        dir_path, file_name = os.path.split(file_path)
        if os.path.isfile(dir_path):
            raise KeyError(key)
        if not os.path.exists(dir_path):
            try:
                os.makedirs(dir_path)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise KeyError(key)

        # write to temporary file
        # note we're not using tempfile.NamedTemporaryFile to avoid restrictive file permissions
        temp_name = file_name + '.' + uuid.uuid4().hex + '.partial'
        temp_path = os.path.join(dir_path, temp_name)
        try:
            self._tofile(value, temp_path)

            # move temporary file into place;
            # make several attempts at writing the temporary file to get past
            # potential antivirus file locking issues
            retry_call(os.replace, (temp_path, file_path), exceptions=(PermissionError,))

        finally:
            # clean up if temp file still exists for whatever reason
            if os.path.exists(temp_path):  # pragma: no cover
                os.remove(temp_path)

    def __delitem__(self, key):
        key = self._normalize_key(key)
        path = os.path.join(self.path, key)
        if os.path.isfile(path):
            os.remove(path)
        elif os.path.isdir(path):
            # include support for deleting directories, even though strictly
            # speaking these do not exist as keys in the store
            shutil.rmtree(path)
        else:
            raise KeyError(key)

    def __contains__(self, key):
        key = self._normalize_key(key)
        file_path = os.path.join(self.path, key)
        return os.path.isfile(file_path)

    def __eq__(self, other):
        return (
            isinstance(other, DirectoryStore) and
            self.path == other.path
        )

    def keys(self):
        if os.path.exists(self.path):
            yield from self._keys_fast(self.path)

    @staticmethod
    def _keys_fast(path, walker=os.walk):
        for dirpath, _, filenames in walker(path):
            dirpath = os.path.relpath(dirpath, path)
            if dirpath == os.curdir:
                for f in filenames:
                    yield f
            else:
                dirpath = dirpath.replace("\\", "/")
                for f in filenames:
                    yield "/".join((dirpath, f))

    def __iter__(self):
        return self.keys()

    def __len__(self):
        return sum(1 for _ in self.keys())

    def dir_path(self, path=None):
        store_path = normalize_storage_path(path)
        dir_path = self.path
        if store_path:
            dir_path = os.path.join(dir_path, store_path)
        return dir_path

    def listdir(self, path=None):
        return self._nested_listdir(path) if self._dimension_separator == "/" else \
            self._flat_listdir(path)

    def _flat_listdir(self, path=None):
        dir_path = self.dir_path(path)
        if os.path.isdir(dir_path):
            return sorted(os.listdir(dir_path))
        else:
            return []

    def _nested_listdir(self, path=None):
        children = self._flat_listdir(path=path)
        if array_meta_key in children:
            # special handling of directories containing an array to map nested chunk
            # keys back to standard chunk keys
            new_children = []
            root_path = self.dir_path(path)
            for entry in children:
                entry_path = os.path.join(root_path, entry)
                if _prog_number.match(entry) and os.path.isdir(entry_path):
                    for dir_path, _, file_names in os.walk(entry_path):
                        for file_name in file_names:
                            file_path = os.path.join(dir_path, file_name)
                            rel_path = file_path.split(root_path + os.path.sep)[1]
                            new_children.append(rel_path.replace(os.path.sep, '.'))
                else:
                    new_children.append(entry)
            return sorted(new_children)
        else:
            return children

    def rename(self, src_path, dst_path):
        store_src_path = normalize_storage_path(src_path)
        store_dst_path = normalize_storage_path(dst_path)

        dir_path = self.path

        src_path = os.path.join(dir_path, store_src_path)
        dst_path = os.path.join(dir_path, store_dst_path)

        os.renames(src_path, dst_path)

    def rmdir(self, path=None):
        store_path = normalize_storage_path(path)
        dir_path = self.path
        if store_path:
            dir_path = os.path.join(dir_path, store_path)
        if os.path.isdir(dir_path):
            shutil.rmtree(dir_path)

    def getsize(self, path=None):
        store_path = normalize_storage_path(path)
        fs_path = self.path
        if store_path:
            fs_path = os.path.join(fs_path, store_path)
        if os.path.isfile(fs_path):
            return os.path.getsize(fs_path)
        elif os.path.isdir(fs_path):
            size = 0
            for child in scandir(fs_path):
                if child.is_file():
                    size += child.stat().st_size
            return size
        else:
            return 0

    def clear(self):
        shutil.rmtree(self.path)


class NestedDirectoryStore(DirectoryStore):
    """Storage class using directories and files on a standard file system, with
    special handling for chunk keys so that chunk files for multidimensional
    arrays are stored in a nested directory tree.

    Parameters
    ----------
    path : string
        Location of directory to use as the root of the storage hierarchy.
    normalize_keys : bool, optional
        If True, all store keys will be normalized to use lower case characters
        (e.g. 'foo' and 'FOO' will be treated as equivalent). This can be
        useful to avoid potential discrepancies between case-sensitive and
        case-insensitive file system. Default value is False.
    dimension_separator : {'/'}, optional
        Separator placed between the dimensions of a chunk.
        Only supports "/" unlike other implementations.

    Examples
    --------
    Store a single array::

        >>> import zarr
        >>> store = zarr.NestedDirectoryStore('data/array.zarr')
        >>> z = zarr.zeros((10, 10), chunks=(5, 5), store=store, overwrite=True)
        >>> z[...] = 42

    Each chunk of the array is stored as a separate file on the file system,
    note the multiple directory levels used for the chunk files::

        >>> import os
        >>> sorted(os.listdir('data/array.zarr'))
        ['.zarray', '0', '1']
        >>> sorted(os.listdir('data/array.zarr/0'))
        ['0', '1']
        >>> sorted(os.listdir('data/array.zarr/1'))
        ['0', '1']

    Store a group::

        >>> store = zarr.NestedDirectoryStore('data/group.zarr')
        >>> root = zarr.group(store=store, overwrite=True)
        >>> foo = root.create_group('foo')
        >>> bar = foo.zeros('bar', shape=(10, 10), chunks=(5, 5))
        >>> bar[...] = 42

    When storing a group, levels in the group hierarchy will correspond to
    directories on the file system, i.e.::

        >>> sorted(os.listdir('data/group.zarr'))
        ['.zgroup', 'foo']
        >>> sorted(os.listdir('data/group.zarr/foo'))
        ['.zgroup', 'bar']
        >>> sorted(os.listdir('data/group.zarr/foo/bar'))
        ['.zarray', '0', '1']
        >>> sorted(os.listdir('data/group.zarr/foo/bar/0'))
        ['0', '1']
        >>> sorted(os.listdir('data/group.zarr/foo/bar/1'))
        ['0', '1']

    Notes
    -----
    The :class:`DirectoryStore` class stores all chunk files for an array
    together in a single directory. On some file systems, the potentially large
    number of files in a single directory can cause performance issues. The
    :class:`NestedDirectoryStore` class provides an alternative where chunk
    files for multidimensional arrays will be organised into a directory
    hierarchy, thus reducing the number of files in any one directory.

    Safe to write in multiple threads or processes.

    """

    def __init__(self, path, normalize_keys=False, dimension_separator="/"):
        super().__init__(path, normalize_keys=normalize_keys)
        if dimension_separator is None:
            dimension_separator = "/"
        elif dimension_separator != "/":
            raise ValueError(
                "NestedDirectoryStore only supports '/' as dimension_separator")
        self._dimension_separator = dimension_separator

    def __eq__(self, other):
        return (
            isinstance(other, NestedDirectoryStore) and
            self.path == other.path
        )


class TempStore(DirectoryStore):
    """Directory store using a temporary directory for storage.

    Parameters
    ----------
    suffix : string, optional
        Suffix for the temporary directory name.
    prefix : string, optional
        Prefix for the temporary directory name.
    dir : string, optional
        Path to parent directory in which to create temporary directory.
    normalize_keys : bool, optional
        If True, all store keys will be normalized to use lower case characters
        (e.g. 'foo' and 'FOO' will be treated as equivalent). This can be
        useful to avoid potential discrepancies between case-sensitive and
        case-insensitive file system. Default value is False.
    dimension_separator : {'.', '/'}, optional
        Separator placed between the dimensions of a chunk.
    """

    # noinspection PyShadowingBuiltins
    def __init__(self, suffix='', prefix='zarr', dir=None, normalize_keys=False,
                 dimension_separator=None):
        path = tempfile.mkdtemp(suffix=suffix, prefix=prefix, dir=dir)
        atexit.register(atexit_rmtree, path)
        super().__init__(path, normalize_keys=normalize_keys)


class DirectoryStoreV3(DirectoryStore, StoreV3):

    def list(self):
        return list(self.keys())

    def __eq__(self, other):
        return (
            isinstance(other, DirectoryStoreV3) and
            self.path == other.path
        )

    def __setitem__(self, key, value):
        self._validate_key(key)
        super().__setitem__(key, value)

    def getsize(self, path: Path = None):
        return _getsize(self, path)

    def rename(self, src_path, dst_path, metadata_key_suffix='.json'):
        store_src_path = normalize_storage_path(src_path)
        store_dst_path = normalize_storage_path(dst_path)

        dir_path = self.path
        any_existed = False
        for root_prefix in ['meta', 'data']:
            src_path = os.path.join(dir_path, root_prefix, 'root', store_src_path)
            if os.path.exists(src_path):
                any_existed = True
                dst_path = os.path.join(dir_path, root_prefix, 'root', store_dst_path)
                os.renames(src_path, dst_path)

        for suffix in ['.array' + metadata_key_suffix,
                       '.group' + metadata_key_suffix]:
            src_meta = os.path.join(dir_path, 'meta', 'root', store_src_path + suffix)
            if os.path.exists(src_meta):
                any_existed = True
                dst_meta = os.path.join(dir_path, 'meta', 'root', store_dst_path + suffix)
                dst_dir = os.path.dirname(dst_meta)
                if not os.path.exists(dst_dir):
                    os.makedirs(dst_dir)
                os.rename(src_meta, dst_meta)
        if not any_existed:
            raise FileNotFoundError("nothing found at src_path")

    def rmdir(self, path=None):
        store_path = normalize_storage_path(path)
        dir_path = self.path
        if store_path:
            for base in [meta_root, data_root]:
                dir_path = os.path.join(dir_path, base + store_path)
                if os.path.isdir(dir_path):
                    shutil.rmtree(dir_path)

            # remove any associated metadata files
            sfx = _get_metadata_suffix(self)
            meta_dir = (meta_root + path).rstrip('/')
            array_meta_file = meta_dir + '.array' + sfx
            self.pop(array_meta_file, None)
            group_meta_file = meta_dir + '.group' + sfx
            self.pop(group_meta_file, None)

        elif os.path.isdir(dir_path):
            shutil.rmtree(dir_path)


DirectoryStoreV3.__doc__ = DirectoryStore.__doc__
