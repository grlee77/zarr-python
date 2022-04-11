"""This module contains storage classes related to MongoDB (pymongo)"""

from numcodecs.compat import ensure_bytes
from zarr._storage.store import Store
from zarr._storage.store_v3 import RmdirV3, StoreV3

__doctest_requires__ = {
    ('MongoDBStore', 'MongoDBStore.*'): ['pymongo'],
    ('MongoDBStoreV3', 'MongoDBStoreV3.*'): ['pymongo'],
}


class MongoDBStore(Store):
    """Storage class using MongoDB.

    .. note:: This is an experimental feature.

    Requires the `pymongo <https://api.mongodb.com/python/current/>`_
    package to be installed.

    Parameters
    ----------
    database : string
        Name of database
    collection : string
        Name of collection
    dimension_separator : {'.', '/'}, optional
        Separator placed between the dimensions of a chunk.
    **kwargs
        Keyword arguments passed through to the `pymongo.MongoClient` function.

    Notes
    -----
    The maximum chunksize in MongoDB documents is 16 MB.

    """

    _key = 'key'
    _value = 'value'

    def __init__(self, database='mongodb_zarr', collection='zarr_collection',
                 dimension_separator=None, **kwargs):
        import pymongo

        self._database = database
        self._collection = collection
        self._dimension_separator = dimension_separator
        self._kwargs = kwargs

        self.client = pymongo.MongoClient(**self._kwargs)
        self.db = self.client.get_database(self._database)
        self.collection = self.db.get_collection(self._collection)

    def __getitem__(self, key):
        doc = self.collection.find_one({self._key: key})

        if doc is None:
            raise KeyError(key)
        else:
            return doc[self._value]

    def __setitem__(self, key, value):
        value = ensure_bytes(value)
        self.collection.replace_one({self._key: key},
                                    {self._key: key, self._value: value},
                                    upsert=True)

    def __delitem__(self, key):
        result = self.collection.delete_many({self._key: key})
        if not result.deleted_count == 1:
            raise KeyError(key)

    def __iter__(self):
        for f in self.collection.find({}):
            yield f[self._key]

    def __len__(self):
        return self.collection.count_documents({})

    def __getstate__(self):
        return self._database, self._collection, self._kwargs

    def __setstate__(self, state):
        database, collection, kwargs = state
        self.__init__(database=database, collection=collection, **kwargs)

    def close(self):
        """Cleanup client resources and disconnect from MongoDB."""
        self.client.close()

    def clear(self):
        """Remove all items from store."""
        self.collection.delete_many({})


class MongoDBStoreV3(RmdirV3, MongoDBStore, StoreV3):

    def list(self):
        return list(self.keys())

    def __setitem__(self, key, value):
        self._validate_key(key)
        super().__setitem__(key, value)


MongoDBStoreV3.__doc__ = MongoDBStore.__doc__
