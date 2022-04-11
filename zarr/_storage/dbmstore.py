"""This module contains storage classes using DBM-style databases"""
import os
from threading import Lock

from numcodecs.compat import ensure_bytes, ensure_text

from zarr._storage.store import _rmdir_from_keys, Store
from zarr._storage.store_v3 import RmdirV3, StoreV3
from zarr.util import nolock, normalize_storage_path


# noinspection PyShadowingBuiltins
class DBMStore(Store):
    """Storage class using a DBM-style database.

    Parameters
    ----------
    path : string
        Location of database file.
    flag : string, optional
        Flags for opening the database file.
    mode : int
        File mode used if a new file is created.
    open : function, optional
        Function to open the database file. If not provided, :func:`dbm.open` will be
        used on Python 3, and :func:`anydbm.open` will be used on Python 2.
    write_lock: bool, optional
        Use a lock to prevent concurrent writes from multiple threads (True by default).
    dimension_separator : {'.', '/'}, optional
        Separator placed between the dimensions of a chunk.e
    **open_kwargs
        Keyword arguments to pass the `open` function.

    Examples
    --------
    Store a single array::

        >>> import zarr
        >>> store = zarr.DBMStore('data/array.db')
        >>> z = zarr.zeros((10, 10), chunks=(5, 5), store=store, overwrite=True)
        >>> z[...] = 42
        >>> store.close()  # don't forget to call this when you're done

    Store a group::

        >>> store = zarr.DBMStore('data/group.db')
        >>> root = zarr.group(store=store, overwrite=True)
        >>> foo = root.create_group('foo')
        >>> bar = foo.zeros('bar', shape=(10, 10), chunks=(5, 5))
        >>> bar[...] = 42
        >>> store.close()  # don't forget to call this when you're done

    After modifying a DBMStore, the ``close()`` method must be called, otherwise
    essential data may not be written to the underlying database file. The
    DBMStore class also supports the context manager protocol, which ensures the
    ``close()`` method is called on leaving the context, e.g.::

        >>> with zarr.DBMStore('data/array.db') as store:
        ...     z = zarr.zeros((10, 10), chunks=(5, 5), store=store, overwrite=True)
        ...     z[...] = 42
        ...     # no need to call store.close()

    A different database library can be used by passing a different function to
    the `open` parameter. For example, if the `bsddb3
    <https://www.jcea.es/programacion/pybsddb.htm>`_ package is installed, a
    Berkeley DB database can be used::

        >>> import bsddb3
        >>> store = zarr.DBMStore('data/array.bdb', open=bsddb3.btopen)
        >>> z = zarr.zeros((10, 10), chunks=(5, 5), store=store, overwrite=True)
        >>> z[...] = 42
        >>> store.close()

    Notes
    -----
    Please note that, by default, this class will use the Python standard
    library `dbm.open` function to open the database file (or `anydbm.open` on
    Python 2). There are up to three different implementations of DBM-style
    databases available in any Python installation, and which one is used may
    vary from one system to another.  Database file formats are not compatible
    between these different implementations.  Also, some implementations are
    more efficient than others. In particular, the "dumb" implementation will be
    the fall-back on many systems, and has very poor performance for some usage
    scenarios. If you want to ensure a specific implementation is used, pass the
    corresponding open function, e.g., `dbm.gnu.open` to use the GNU DBM
    library.

    Safe to write in multiple threads. May be safe to write in multiple processes,
    depending on which DBM implementation is being used, although this has not been
    tested.

    """

    def __init__(self, path, flag='c', mode=0o666, open=None, write_lock=True,
                 dimension_separator=None,
                 **open_kwargs):
        if open is None:
            import dbm
            open = dbm.open
        path = os.path.abspath(path)
        # noinspection PyArgumentList
        self.db = open(path, flag, mode, **open_kwargs)
        self.path = path
        self.flag = flag
        self.mode = mode
        self.open = open
        self.write_lock = write_lock
        if write_lock:
            # This may not be required as some dbm implementations manage their own
            # locks, but err on the side of caution.
            self.write_mutex = Lock()
        else:
            self.write_mutex = nolock
        self.open_kwargs = open_kwargs
        self._dimension_separator = dimension_separator

    def __getstate__(self):
        try:
            self.flush()  # needed for ndbm
        except Exception:
            # flush may fail if db has already been closed
            pass
        return (self.path, self.flag, self.mode, self.open, self.write_lock,
                self.open_kwargs)

    def __setstate__(self, state):
        path, flag, mode, open, write_lock, open_kws = state
        if flag[0] == 'n':
            flag = 'c' + flag[1:]  # don't clobber an existing database
        self.__init__(path=path, flag=flag, mode=mode, open=open,
                      write_lock=write_lock, **open_kws)

    def close(self):
        """Closes the underlying database file."""
        if hasattr(self.db, 'close'):
            with self.write_mutex:
                self.db.close()

    def flush(self):
        """Synchronizes data to the underlying database file."""
        if self.flag[0] != 'r':
            with self.write_mutex:
                if hasattr(self.db, 'sync'):
                    self.db.sync()
                else:  # pragma: no cover
                    # we don't cover this branch anymore as ndbm (oracle) is not packaged
                    # by conda-forge on non-mac OS:
                    # https://github.com/conda-forge/staged-recipes/issues/4476
                    # fall-back, close and re-open, needed for ndbm
                    flag = self.flag
                    if flag[0] == 'n':
                        flag = 'c' + flag[1:]  # don't clobber an existing database
                    self.db.close()
                    # noinspection PyArgumentList
                    self.db = self.open(self.path, flag, self.mode, **self.open_kwargs)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __getitem__(self, key):
        if isinstance(key, str):
            key = key.encode("ascii")
        return self.db[key]

    def __setitem__(self, key, value):
        if isinstance(key, str):
            key = key.encode("ascii")
        value = ensure_bytes(value)
        with self.write_mutex:
            self.db[key] = value

    def __delitem__(self, key):
        if isinstance(key, str):
            key = key.encode("ascii")
        with self.write_mutex:
            del self.db[key]

    def __eq__(self, other):
        return (
            isinstance(other, DBMStore) and
            self.path == other.path and
            # allow flag and mode to differ
            self.open == other.open and
            self.open_kwargs == other.open_kwargs
        )

    def keys(self):
        return (ensure_text(k, "ascii") for k in iter(self.db.keys()))

    def __iter__(self):
        return self.keys()

    def __len__(self):
        return sum(1 for _ in self.keys())

    def __contains__(self, key):
        if isinstance(key, str):
            key = key.encode("ascii")
        return key in self.db

    def rmdir(self, path: str = "") -> None:
        path = normalize_storage_path(path)
        _rmdir_from_keys(self, path)


class DBMStoreV3(RmdirV3, DBMStore, StoreV3):

    def list(self):
        return list(self.keys())

    def __setitem__(self, key, value):
        self._validate_key(key)
        super().__setitem__(key, value)


DBMStoreV3.__doc__ = DBMStore.__doc__
