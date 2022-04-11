"""This module contains storage classes using LMDB"""
import multiprocessing
import os
import sys

from numcodecs.compat import ensure_text

from zarr._storage.store import Store
from zarr._storage.store_v3 import RmdirV3, StoreV3


class LMDBStore(Store):
    """Storage class using LMDB. Requires the `lmdb <http://lmdb.readthedocs.io/>`_
    package to be installed.


    Parameters
    ----------
    path : string
        Location of database file.
    buffers : bool, optional
        If True (default) use support for buffers, which should increase performance by
        reducing memory copies.
    dimension_separator : {'.', '/'}, optional
        Separator placed between the dimensions of a chunk.
    **kwargs
        Keyword arguments passed through to the `lmdb.open` function.

    Examples
    --------
    Store a single array::

        >>> import zarr
        >>> store = zarr.LMDBStore('data/array.mdb')
        >>> z = zarr.zeros((10, 10), chunks=(5, 5), store=store, overwrite=True)
        >>> z[...] = 42
        >>> store.close()  # don't forget to call this when you're done

    Store a group::

        >>> store = zarr.LMDBStore('data/group.mdb')
        >>> root = zarr.group(store=store, overwrite=True)
        >>> foo = root.create_group('foo')
        >>> bar = foo.zeros('bar', shape=(10, 10), chunks=(5, 5))
        >>> bar[...] = 42
        >>> store.close()  # don't forget to call this when you're done

    After modifying a DBMStore, the ``close()`` method must be called, otherwise
    essential data may not be written to the underlying database file. The
    DBMStore class also supports the context manager protocol, which ensures the
    ``close()`` method is called on leaving the context, e.g.::

        >>> with zarr.LMDBStore('data/array.mdb') as store:
        ...     z = zarr.zeros((10, 10), chunks=(5, 5), store=store, overwrite=True)
        ...     z[...] = 42
        ...     # no need to call store.close()

    Notes
    -----
    By default writes are not immediately flushed to disk to increase performance. You
    can ensure data are flushed to disk by calling the ``flush()`` or ``close()`` methods.

    Should be safe to write in multiple threads or processes due to the synchronization
    support within LMDB, although writing from multiple processes has not been tested.

    """

    def __init__(self, path, buffers=True, dimension_separator=None, **kwargs):
        import lmdb

        # set default memory map size to something larger than the lmdb default, which is
        # very likely to be too small for any moderate array (logic copied from zict)
        map_size = (2**40 if sys.maxsize >= 2**32 else 2**28)
        kwargs.setdefault('map_size', map_size)

        # don't initialize buffers to zero by default, shouldn't be necessary
        kwargs.setdefault('meminit', False)

        # decide whether to use the writemap option based on the operating system's
        # support for sparse files - writemap requires sparse file support otherwise
        # the whole# `map_size` may be reserved up front on disk (logic copied from zict)
        writemap = sys.platform.startswith('linux')
        kwargs.setdefault('writemap', writemap)

        # decide options for when data are flushed to disk - choose to delay syncing
        # data to filesystem, otherwise pay a large performance penalty (zict also does
        # this)
        kwargs.setdefault('metasync', False)
        kwargs.setdefault('sync', False)
        kwargs.setdefault('map_async', False)

        # set default option for number of cached transactions
        max_spare_txns = multiprocessing.cpu_count()
        kwargs.setdefault('max_spare_txns', max_spare_txns)

        # normalize path
        path = os.path.abspath(path)

        # open database
        self.db = lmdb.open(path, **kwargs)

        # store properties
        self.buffers = buffers
        self.path = path
        self.kwargs = kwargs
        self._dimension_separator = dimension_separator

    def __getstate__(self):
        try:
            self.flush()  # just in case
        except Exception:
            # flush may fail if db has already been closed
            pass
        return self.path, self.buffers, self.kwargs

    def __setstate__(self, state):
        path, buffers, kwargs = state
        self.__init__(path=path, buffers=buffers, **kwargs)

    def close(self):
        """Closes the underlying database."""
        self.db.close()

    def flush(self):
        """Synchronizes data to the file system."""
        self.db.sync()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __getitem__(self, key):
        if isinstance(key, str):
            key = key.encode("ascii")
        # use the buffers option, should avoid a memory copy
        with self.db.begin(buffers=self.buffers) as txn:
            value = txn.get(key)
        if value is None:
            raise KeyError(key)
        return value

    def __setitem__(self, key, value):
        if isinstance(key, str):
            key = key.encode("ascii")
        with self.db.begin(write=True, buffers=self.buffers) as txn:
            txn.put(key, value)

    def __delitem__(self, key):
        if isinstance(key, str):
            key = key.encode("ascii")
        with self.db.begin(write=True) as txn:
            if not txn.delete(key):
                raise KeyError(key)

    def __contains__(self, key):
        if isinstance(key, str):
            key = key.encode("ascii")
        with self.db.begin(buffers=self.buffers) as txn:
            with txn.cursor() as cursor:
                return cursor.set_key(key)

    def items(self):
        with self.db.begin(buffers=self.buffers) as txn:
            with txn.cursor() as cursor:
                for k, v in cursor.iternext(keys=True, values=True):
                    yield ensure_text(k, "ascii"), v

    def keys(self):
        with self.db.begin(buffers=self.buffers) as txn:
            with txn.cursor() as cursor:
                for k in cursor.iternext(keys=True, values=False):
                    yield ensure_text(k, "ascii")

    def values(self):
        with self.db.begin(buffers=self.buffers) as txn:
            with txn.cursor() as cursor:
                for v in cursor.iternext(keys=False, values=True):
                    yield v

    def __iter__(self):
        return self.keys()

    def __len__(self):
        return self.db.stat()['entries']


class LMDBStoreV3(RmdirV3, LMDBStore, StoreV3):

    def list(self):
        return list(self.keys())

    def __setitem__(self, key, value):
        self._validate_key(key)
        super().__setitem__(key, value)


LMDBStoreV3.__doc__ = LMDBStore.__doc__
