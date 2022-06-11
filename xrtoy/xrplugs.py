import dask.array as da
import h5py
import numpy as np
import xarray as xr
from natsort import natsorted
from xarray.backends import AbstractDataStore, BackendArray, BackendEntrypoint
from xarray.backends.locks import HDF5_LOCK
from xarray.core import indexing
from xarray.core.utils import Frozen

from .data import _is_valid_xrt_file, read_data


class WrappedXRTVariable:
    def __init__(
        self,
        dimensions,
        data,
        attributes,
    ):
        self.dimensions = dimensions
        self.data = data
        self.attributes = attributes


class WrappedXRTDataset:
    def __init__(
        self,
        filename_or_obj,
        mode="r",
    ):
        self._raw_coords, self._raw_data = read_data(filename_or_obj)
        self._file = h5py.File(filename_or_obj, mode=mode)
        dimensions, variables, attributes, encoding = self._read_file(self._file)

        self.dimensions = dimensions
        self.variables = variables
        self.attributes = attributes
        self.encoding = encoding

    def _read_file(self, file_handle):
        encoding = {}
        variables = {}
        attributes = {}

        dimensions = tuple(self._raw_coords.keys())
        dimensions = ("time",) + tuple(
            natsorted([d for d in dimensions if d != "time"])
        )
        var_data = da.stack([file_handle[daddr[1]] for daddr in self._raw_data])
        variables["myvar"] = WrappedXRTVariable(
            dimensions,
            var_data,
            attributes,
        )

        variables["time"] = WrappedXRTVariable(("time",), file_handle["time"], {})
        for name, value in self._raw_coords.items():
            variables[name] = WrappedXRTVariable((name,), value, {})

        return dimensions, variables, attributes, encoding

    def close(self):
        self._file.close()

    __del__ = close


class XRTDataStore(AbstractDataStore):
    def __init__(
        self,
        filename_or_obj,
        lock=HDF5_LOCK,
        **backend_kwargs,
    ):
        self.lock = xr.backends.locks.ensure_lock(lock)

        self._manager = xr.backends.CachingFileManager(
            WrappedXRTDataset,
            filename_or_obj,
            mode="r",
            lock=self.lock,
        )
        self._backend_kwargs = backend_kwargs

    def _acquire(self, needs_lock=True):
        with self._manager.acquire_context(needs_lock) as root:
            ds = root
        return ds

    @property
    def ds(self):
        return self._acquire()

    def open_store_variable(self, name, var):
        if isinstance(var.data, np.ndarray):
            data = var.data
        else:
            wrapped_array = XRTArrayWrapper(self, name)
            data = xr.core.indexing.LazilyIndexedArray(wrapped_array)

        encoding = self.ds.encoding.copy()
        return xr.Variable(var.dimensions, data, var.attributes, encoding)

    def get_dimensions(self):
        return xr.core.utils.Frozen(self.ds.dimensions)

    def get_attrs(self):
        return Frozen(self.ds.attributes)

    def get_variables(self):
        return xr.core.utils.FrozenDict(
            (k, self.open_store_variable(k, v)) for k, v in self.ds.variables.items()
        )

    def close(self, **kwargs):
        self._manager.close(**kwargs)


class XRTBackendEntrypoint(BackendEntrypoint):
    def open_dataset(
        self,
        filename_or_obj,
        *,
        drop_variables=None,
    ):
        store = XRTDataStore(filename_or_obj)

        with xr.core.utils.close_on_error(store):
            variables, attrs = store.load()
            # decode...
            ds = xr.Dataset(
                variables,
                attrs=attrs,
                # coords=coords,
            )
            ds.set_coords([k for k in ds if k.startswith("coord") or k == "time"])
            ds.set_close(store.close)
        return ds

    open_dataset_parameters = ["filename_or_obj", "drop_variables"]

    def guess_can_open(self, filename_or_obj):
        return _is_valid_xrt_file(filename_or_obj)


class XRTArrayWrapper(BackendArray):
    def __init__(
        self,
        datastore,
        variable_name,
    ):
        self.datastore = datastore
        self.variable_name = variable_name
        array = self.get_array()
        self.shape = array.data.shape
        self.dtype = array.data.dtype

    def get_array(self, needs_lock=True):
        ds = self.datastore._acquire(needs_lock)
        variable = ds.variables[self.variable_name]
        return variable

    def __getitem__(self, key):
        return indexing.explicit_indexing_adapter(
            key,
            self.shape,
            indexing.IndexingSupport.OUTER,
            self._getitem,
        )

    def _getitem(self, key):
        # A thread safe method (that access to data on disk) is required for
        # distributed computing
        with self.datastore.lock:
            # Locking the datastore is not enough! Need to ensure store lock
            # /and/ call its acquire in case the desired file goes out of cache
            array = self.get_array(needs_lock=False)
            return array.data[key]
