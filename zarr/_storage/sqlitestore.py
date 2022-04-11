"""This module contains storage classes using SQLite"""
import operator
import os
from pickle import PicklingError
from threading import Lock

from numcodecs.compat import ensure_contiguous_ndarray

from zarr._storage.store import _getsize, Store
from zarr._storage.store_v3 import _get_metadata_suffix, data_root, meta_root, StoreV3
from zarr.util import normalize_storage_path


class SQLiteStore(Store):
    """Storage class using SQLite.

    Parameters
    ----------
    path : string
        Location of database file.
    dimension_separator : {'.', '/'}, optional
        Separator placed between the dimensions of a chunk.
    **kwargs
        Keyword arguments passed through to the `sqlite3.connect` function.

    Examples
    --------
    Store a single array::

        >>> import zarr
        >>> store = zarr.SQLiteStore('data/array.sqldb')
        >>> z = zarr.zeros((10, 10), chunks=(5, 5), store=store, overwrite=True)
        >>> z[...] = 42
        >>> store.close()  # don't forget to call this when you're done

    Store a group::

        >>> store = zarr.SQLiteStore('data/group.sqldb')
        >>> root = zarr.group(store=store, overwrite=True)
        >>> foo = root.create_group('foo')
        >>> bar = foo.zeros('bar', shape=(10, 10), chunks=(5, 5))
        >>> bar[...] = 42
        >>> store.close()  # don't forget to call this when you're done
    """

    def __init__(self, path, dimension_separator=None, **kwargs):
        import sqlite3

        self._dimension_separator = dimension_separator

        # normalize path
        if path != ':memory:':
            path = os.path.abspath(path)

        # store properties
        self.path = path
        self.kwargs = kwargs

        # allow threading if SQLite connections are thread-safe
        #
        # ref: https://www.sqlite.org/releaselog/3_3_1.html
        # ref: https://bugs.python.org/issue27190
        check_same_thread = True
        if sqlite3.sqlite_version_info >= (3, 3, 1):
            check_same_thread = False

        # keep a lock for serializing mutable operations
        self.lock = Lock()

        # open database
        self.db = sqlite3.connect(
            self.path,
            detect_types=0,
            isolation_level=None,
            check_same_thread=check_same_thread,
            **self.kwargs
        )

        # handle keys as `str`s
        self.db.text_factory = str

        # get a cursor to read/write to the database
        self.cursor = self.db.cursor()

        # initialize database with our table if missing
        with self.lock:
            self.cursor.execute(
                'CREATE TABLE IF NOT EXISTS zarr(k TEXT PRIMARY KEY, v BLOB)'
            )

    def __getstate__(self):
        if self.path == ':memory:':
            raise PicklingError('Cannot pickle in-memory SQLite databases')
        return self.path, self.kwargs

    def __setstate__(self, state):
        path, kwargs = state
        self.__init__(path=path, **kwargs)

    def close(self):
        """Closes the underlying database."""

        # close cursor and db objects
        self.cursor.close()
        self.db.close()

    def __getitem__(self, key):
        value = self.cursor.execute('SELECT v FROM zarr WHERE (k = ?)', (key,))
        for v, in value:
            return v
        raise KeyError(key)

    def __setitem__(self, key, value):
        self.update({key: value})

    def __delitem__(self, key):
        with self.lock:
            self.cursor.execute('DELETE FROM zarr WHERE (k = ?)', (key,))
            if self.cursor.rowcount < 1:
                raise KeyError(key)

    def __contains__(self, key):
        cs = self.cursor.execute(
            'SELECT COUNT(*) FROM zarr WHERE (k = ?)', (key,)
        )
        for has, in cs:
            has = bool(has)
            return has

    def items(self):
        kvs = self.cursor.execute('SELECT k, v FROM zarr')
        for k, v in kvs:
            yield k, v

    def keys(self):
        ks = self.cursor.execute('SELECT k FROM zarr')
        for k, in ks:
            yield k

    def values(self):
        vs = self.cursor.execute('SELECT v FROM zarr')
        for v, in vs:
            yield v

    def __iter__(self):
        return self.keys()

    def __len__(self):
        cs = self.cursor.execute('SELECT COUNT(*) FROM zarr')
        for c, in cs:
            return c

    def update(self, *args, **kwargs):
        args += (kwargs,)

        kv_list = []
        for dct in args:
            for k, v in dct.items():
                v = ensure_contiguous_ndarray(v)

                # Accumulate key-value pairs for storage
                kv_list.append((k, v))

        with self.lock:
            self.cursor.executemany('REPLACE INTO zarr VALUES (?, ?)', kv_list)

    def listdir(self, path=None):
        path = normalize_storage_path(path)
        sep = '_' if path == '' else '/'
        keys = self.cursor.execute(
            '''
            SELECT DISTINCT SUBSTR(m, 0, INSTR(m, "/")) AS l FROM (
                SELECT LTRIM(SUBSTR(k, LENGTH(?) + 1), "/") || "/" AS m
                FROM zarr WHERE k LIKE (? || "{sep}%")
            ) ORDER BY l ASC
            '''.format(sep=sep),
            (path, path)
        )
        keys = list(map(operator.itemgetter(0), keys))
        return keys

    def getsize(self, path=None):
        path = normalize_storage_path(path)
        size = self.cursor.execute(
            '''
            SELECT COALESCE(SUM(LENGTH(v)), 0) FROM zarr
            WHERE k LIKE (? || "%") AND
                  0 == INSTR(LTRIM(SUBSTR(k, LENGTH(?) + 1), "/"), "/")
            ''',
            (path, path)
        )
        for s, in size:
            return s

    def rmdir(self, path=None):
        path = normalize_storage_path(path)
        if path:
            with self.lock:
                self.cursor.execute(
                    'DELETE FROM zarr WHERE k LIKE (? || "/%")', (path,)
                )
        else:
            self.clear()

    def clear(self):
        with self.lock:
            self.cursor.executescript(
                '''
                BEGIN TRANSACTION;
                    DROP TABLE zarr;
                    CREATE TABLE zarr(k TEXT PRIMARY KEY, v BLOB);
                COMMIT TRANSACTION;
                '''
            )


class SQLiteStoreV3(SQLiteStore, StoreV3):

    def list(self):
        return list(self.keys())

    def getsize(self, path=None):
        # TODO: why does the query below not work in this case?
        #       For now fall back to the default _getsize implementation
        # size = 0
        # for _path in [data_root + path, meta_root + path]:
        #     c = self.cursor.execute(
        #         '''
        #         SELECT COALESCE(SUM(LENGTH(v)), 0) FROM zarr
        #         WHERE k LIKE (? || "%") AND
        #               0 == INSTR(LTRIM(SUBSTR(k, LENGTH(?) + 1), "/"), "/")
        #         ''',
        #         (_path, _path)
        #     )
        #     for item_size, in c:
        #         size += item_size
        # return size

        # fallback to default implementation for now
        return _getsize(self, path)

    def __setitem__(self, key, value):
        self._validate_key(key)
        super().__setitem__(key, value)

    def rmdir(self, path=None):
        path = normalize_storage_path(path)
        if path:
            for base in [meta_root, data_root]:
                with self.lock:
                    self.cursor.execute(
                        'DELETE FROM zarr WHERE k LIKE (? || "/%")', (base + path,)
                    )
            # remove any associated metadata files
            sfx = _get_metadata_suffix(self)
            meta_dir = (meta_root + path).rstrip('/')
            array_meta_file = meta_dir + '.array' + sfx
            self.pop(array_meta_file, None)
            group_meta_file = meta_dir + '.group' + sfx
            self.pop(group_meta_file, None)
        else:
            self.clear()


SQLiteStoreV3.__doc__ = SQLiteStore.__doc__
