# Toy model to implementing Xarray backends

This repository explores how to implement [Xarray](https://xarray.pydata.org)
backends, thus allowing to read arbitrarily formatted data.  The code is
focused on a parallelizable version of the backends with Dask, as well as HDF5
data formats, which require complex reading operations (such as concatenation
of multiple datasets into a single DataArray).
