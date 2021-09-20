from collections.abc import MutableMapping
from typing import Optional, Union, List, Tuple, Dict, Any

from zarr.meta import Metadata2, Metadata3, _default_entry_point_metadata_v3
from zarr.util import normalize_storage_path

# v2 store keys
array_meta_key = '.zarray'
group_meta_key = '.zgroup'
attrs_key = '.zattrs'


class Store(MutableMapping):
    """Base class for stores implementation.

    Provide a number of default method as well as other typing guaranties for
    mypy.

    Stores cannot be mutable mapping as they do have a couple of other
    requirements that would break Liskov substitution principle (stores only
    allow strings as keys, mutable mapping are more generic).

    And Stores do requires a few other method.

    Having no-op base method also helps simplifying store usage and do not need
    to check the presence of attributes and methods, like `close()`.

    Stores can be used as context manager to make sure they close on exit.

    .. added: 2.11.0

    """

    _readable = True
    _writeable = True
    _erasable = True
    _listable = True
    _store_version = 2           # v2-specific
    _metadata_class = Metadata2  # v2-specific
    # TODO: add _dimension_separator to Store? would require updating hashes again in test_core.py

    def is_readable(self):
        return self._readable

    def is_writeable(self):
        return self._writeable

    def is_listable(self):
        return self._listable

    def is_erasable(self):
        return self._erasable

    def __enter__(self):
        if not hasattr(self, "_open_count"):
            self._open_count = 0
        self._open_count += 1
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._open_count -= 1
        if self._open_count == 0:
            self.close()

    # TODO: existing zarr-python v2 stores define listdir, but the v3 spec
    #       calls this method list_dir.
    def listdir(self, path: str = "") -> List[str]:
        path = normalize_storage_path(path)
        return _listdir_from_keys(self, path)

    def rename(self, src_path: str, dst_path: str) -> None:
        if not self.is_erasable():
            raise NotImplementedError(
                f'{type(self)} is not erasable, cannot call "rename"'
            )  # pragma: no cover
        _rename_from_keys(self, src_path, dst_path)

    def rmdir(self, path: str = "") -> None:
        if not self.is_erasable():
            raise NotImplementedError(
                f'{type(self)} is not erasable, cannot call "rmdir"'
            )  # pragma: no cover
        path = normalize_storage_path(path)
        _rmdir_from_keys(self, path)

    def close(self) -> None:
        """Do nothing by default"""
        pass

    @staticmethod
    def _ensure_store(store):
        """
        We want to make sure internally that zarr stores are always a class
        with a specific interface derived from ``Store``, which is slightly
        different than ``MutableMapping``.

        We'll do this conversion in a few places automatically
        """
        from zarr.storage import KVStore  # avoid circular import

        if store is None:
            return None
        elif isinstance(store, Store):
            return store
        elif isinstance(store, MutableMapping):
            return KVStore(store)
        else:
            for attr in [
                "keys",
                "values",
                "get",
                "__setitem__",
                "__getitem__",
                "__delitem__",
                "__contains__",
            ]:
                if not hasattr(store, attr):
                    break
            else:
                return KVStore(store)

        raise ValueError(
            "Starting with Zarr 2.9.0, stores must be subclasses of Store, if "
            "your store exposes the MutableMapping interface wrap it in "
            f"Zarr.storage.KVStore. Got {store}"
        )



class StoreV3(Store):
    _store_version = 3
    _metadata_class = Metadata3

    @staticmethod
    def _valid_key(key: str) -> bool:
        """
        Verify that a key conforms to the specification.

        A key is any string containing only character in the range a-z, A-Z,
        0-9, or in the set /.-_ it will return True if that's the case, False
        otherwise.

        In addition, in spec v3, keys can only start with the prefix meta/,
        data/ or be exactly zarr.json and should not end with /. This should
        not be exposed to the user, and is a store implementation detail, so
        this method will raise a ValueError in that case.
        """
        if sys.version_info > (3, 7):
            if not key.isascii():
                return False
        if set(key) - set(ascii_letters + digits + "/.-_"):
            return False

        if (
            not key.startswith("data/")
            and (not key.startswith("meta/"))
            and (not key == "zarr.json")
        ):
            raise ValueError("keys starts with unexpected value: `{}`".format(key))

        if key.endswith('/'):
            raise ValueError("keys may not end in /")

        return True

    # Disable get method. DirectoryStoreV3 tests in test_core_v3.py use
    #     .get(key, default) syntax within Array.hexdigest which fails if this
    #     get overrides the get method of MutableMapping

    # def get(self, key: str):
    #     """
    #     default implementation of get that validate the key, a
    #     check that the return value by bytes. This relies on ``def _get(key)``
    #     to be implmented.

    #     Will ensure that the following are correct:
    #         - return group metadata objects are json and contain a single
    #           `attributes` key.
    #     """
    #     assert self._valid_key(key), key
    #     result = self._get(key)
    #     assert isinstance(result, bytes), "Expected bytes, got {}".format(result)
    #     if key == "zarr.json":
    #         v = json.loads(result.decode())
    #         assert set(v.keys()) == {
    #             "zarr_format",
    #             "metadata_encoding",
    #             "metadata_key_suffix",
    #             "extensions",
    #         }, "v is {}".format(v)
    #     elif key.endswith("/.group"):
    #         v = json.loads(result.decode())
    #         assert set(v.keys()) == {"attributes"}, "got unexpected keys {}".format(
    #             v.keys()
    #         )
    #     return result

    def set(self, key: str, value: bytes):
            """
            default implementation of set that validates the key, and
            checks that the return value by bytes. This relies on
            `def _set(key, value)` to be implmented.

            Will ensure that the following are correct:
                - set group metadata objects are json and contain a single
                  `attributes` key.
            """
            if key == "zarr.json":
                v = json.loads(value.decode())
                assert set(v.keys()) == {
                    "zarr_format",
                    "metadata_encoding",
                    "metadata_key_suffix",
                    "extensions",
                }, "v is {}".format(v)
            elif key.endswith(".array"):
                v = json.loads(value.decode())
                expected = {
                    "shape",
                    "data_type",
                    "chunk_grid",
                    "chunk_memory_layout",
                    "compressor",
                    "fill_value",
                    "extensions",
                    "attributes",
                }
                current = set(v.keys())
                current = current - {'dimension_separator'}  # TODO: ignore possible extra dimension_separator entry in .array
                # ets do some conversions.
                assert current == expected, "{} extra, {} missing in {}".format(
                    current - expected, expected - current, v
                )

            assert isinstance(value, bytes)
            assert self._valid_key(key)
            self._set(key, value)

    def list_prefix(self, prefix):
        if prefix.startswith('/'):
            raise ValueError("prefix must not begin with /")
        # TODO: force prefix to end with /?
        return [k for k in self.list() if k.startswith(prefix)]

    def erase(self, key):
        self.__delitem__(key)

    #def erase(self, key):
    #    del self._mutable_mapping[key]

    def erase_prefix(self, prefix):
        assert prefix.endswith("/")

        if prefix == "/":
            all_keys = self.list()
        else:
            all_keys = self.list_prefix(prefix)
        for key in all_keys:
            self.erase(key)

    # TODO: what was this for? (was in Matthias's v3 branch)
    # def initialize(self):
    #     pass

    def list_dir(self, prefix):
        """
        Note: carefully test this with trailing/leading slashes
        """
        if prefix:  # allow prefix = "" ?
            assert prefix.endswith("/")

        all_keys = self.list_prefix(prefix)
        len_prefix = len(prefix)
        keys = []
        prefixes = []
        for k in all_keys:
            trail = k[len_prefix:]
            if "/" not in trail:
                keys.append(prefix + trail)
            else:
                prefixes.append(prefix + trail.split("/", maxsplit=1)[0] + "/")
        return keys, list(set(prefixes))


    def list(self):
        if hasattr(self, 'keys'):
            return list(self.keys())
        raise NotImplementedError(
            "The list method has not been implemented for this store type."
        )

    # Remove? This method is just to match the current V2 stores
    # The v3 spec mentions: list, list_dir, list_prefix
    def listdir(self, path: str = ""):  # to override inherited v2 listdir
        # TODO: just call list_dir or raise NotImpelementedError?
        if path and not path.endswith("/"):
            path = path + "/"
        keys, prefixes = self.list_dir(path)
        prefixes = [p[len(path):].rstrip("/") for p in prefixes]
        keys = [k[len(path):] for k in keys]
        return keys + prefixes

    # TODO: this was in Matthias's branch, but may want to just keep __contains__ only
    #def contains(self, key):
    #    assert key.startswith(("meta/", "data/")), "Got {}".format(key)
    #    return key in self.list()

    def __contains__(self, key):
        # TODO: re-enable this check?
        # if not key.startswith(("meta/", "data/")):
        #     raise ValueError(
        #         f'Key must start with either "meta/" or "data/". '
        #         f'Got {key}'
        #     )
        return key in self.list()

    def clear(self):
        """Remove all items from store."""
        self.erase_prefix("/")

    #def __eq__(self, other):
    #    return type(other) == type(self) and self.path == other.path

    def __eq__(self, other):
        from zarr.storage import KVStoreV3  # avoid circular import
        if isinstance(other, KVStoreV3):
            return self._mutable_mapping == other._mutable_mapping
        else:
            return NotImplemented

    @staticmethod
    def _ensure_store(store):
        """
        We want to make sure internally that zarr stores are always a class
        with a specific interface derived from ``Store``, which is slightly
        different than ``MutableMapping``.

        We'll do this conversion in a few places automatically
        """
        from zarr.storage import KVStoreV3  # avoid circular import
        if store is None:
            return None
        elif isinstance(store, Store):
            return store
        elif isinstance(store, MutableMapping):
            return KVStoreV3(store)
        else:
            for attr in [
                "keys",
                "values",
                "get",
                "__setitem__",
                "__getitem__",
                "__delitem__",
                "__contains__",
            ]:
                if not hasattr(store, attr):
                    break
            else:
                return KVStoreV3(store)

        raise ValueError(
            "Starting with Zarr 2.9.0, stores must be subclasses of Store, if "
            "your store exposes the MutableMapping interface wrap it in "
            f"Zarr.storage.KVStoreV3. Got {store}"
        )


def _path_to_prefix(path: Optional[str]) -> str:
    # assume path already normalized
    if path:
        prefix = path + '/'
    else:
        prefix = ''
    return prefix


# TODO: Should this return default metadata or raise an Error if zarr.json
#       is absent?
def _get_hierarchy_metadata(store=None):
    meta = _default_entry_point_metadata_v3
    if store is not None:
        version = getattr(store, '_store_version', 2)
        if version < 3:
            raise ValueError("zarr.json hierarchy metadata not stored for "
                             f"zarr v{version} stores")
        if 'zarr.json' in store:
            meta = store._metadata_class.decode_hierarchy_metadata(store['zarr.json'])
    return meta


def _rename_from_keys(store: Store, src_path: str, dst_path: str) -> None:
    # assume path already normalized
    src_prefix = _path_to_prefix(src_path)
    dst_prefix = _path_to_prefix(dst_path)
    version = getattr(store, '_store_version', 2)
    if version == 2:
        root_prefixes = ['']
    elif version == 3:
        root_prefixes = ['meta/root/', 'data/root/']
    for root_prefix in root_prefixes:
        _src_prefix = root_prefix + src_prefix
        _dst_prefix = root_prefix + dst_prefix
        for key in list(store.keys()):
            if key.startswith(_src_prefix):
                new_key = _dst_prefix + key.lstrip(_src_prefix)
                store[new_key] = store.pop(key)
    if version == 3:
        sfx = _get_hierarchy_metadata(store)['metadata_key_suffix']
        _src_array_json = 'meta/root/' + src_prefix[:-1] + '.array' + sfx
        if _src_array_json in store:
            new_key = 'meta/root/' + dst_prefix[:-1] + '.array' + sfx
            store[new_key] = store.pop(_src_array_json)
        _src_group_json = 'meta/root/' + src_prefix[:-1] + '.group' + sfx
        if _src_group_json in store:
            new_key = 'meta/root/' + dst_prefix[:-1] + '.group' + sfx
            store[new_key] = store.pop(_src_group_json)


def _rmdir_from_keys(store: Store, path: Optional[str] = None) -> None:
    if getattr(store, '_store_version', 2) == 3:
        prefix = _path_to_prefix(path)
        for key in list(store.keys()):
            if key.startswith(prefix):
                del store[key]
    else:
        # assume path already normalized
        prefix = _path_to_prefix(path)
        for key in list(store.keys()):
            if key.startswith(prefix):
                del store[key]


def _listdir_from_keys(store: Store, path: Optional[str] = None) -> List[str]:
    # assume path already normalized
    prefix = _path_to_prefix(path)
    children = set()
    for key in list(store.keys()):
        if key.startswith(prefix) and len(key) > len(prefix):
            suffix = key[len(prefix):]
            child = suffix.split('/')[0]
            children.add(child)
    return sorted(children)


# TODO: build into __contains__
def _prefix_to_array_key(store: Store, prefix: str) -> str:
    if getattr(store, "_store_version", 2) == 3:
        if prefix:
            sfx = _get_hierarchy_metadata(store)['metadata_key_suffix']
            key = "meta/root/" + prefix.rstrip("/") + ".array" + sfx
        else:
            raise ValueError("prefix must be supplied to get a v3 array key")
    else:
        key = prefix + array_meta_key
    return key


def _prefix_to_group_key(store: Store, prefix: str) -> str:
    if getattr(store, "_store_version", 2) == 3:
        if prefix:
            sfx = _get_hierarchy_metadata(store)['metadata_key_suffix']
            key = "meta/root/" + prefix.rstrip('/') + ".group" + sfx
        else:
            raise ValueError("prefix must be supplied to get a v3 group key")
    else:
        key = prefix + group_meta_key
    return key


def _prefix_to_attrs_key(store: Store, prefix: str) -> str:
    if getattr(store, "_store_version", 2) == 3:
        # for v3, attributes are stored in the array metadata
        sfx = _get_hierarchy_metadata(store)['metadata_key_suffix']
        if prefix:
            key = "meta/root/" + prefix.rstrip('/') + ".array" + sfx
        else:
            raise ValueError("prefix must be supplied to get a v3 array key")
    else:
        key = prefix + attrs_key
    return key
