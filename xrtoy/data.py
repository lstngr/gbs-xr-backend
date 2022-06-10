import dask.array as da
import h5py
import numpy as np
from natsort import natsorted


def save_data(
    base_filename, frame_shape, n_frames, n_files=1, use_dask=False, mode="x"
):
    """
    Generates random data in our "in-house" format

    Saves a series of files containing random data to the disk. The output
    file(s) are HDF5 encoded, and contain a "time" series of arrays with
    requested shape, saved as separate datasets.

    Parameters
    ----------
    base_filename : str
        Naming to use for the output files. For example,
        `base_filename="/tmp/myfile"` will generate a series of files named
        `/tmp/myfile0.xrt`, `/tmp/myfile1.xrt`, ...
    frame_shape : tuple of int
        A valid array shape (accepted by numpy or dask)
    n_frames : int
        The total number of frames desired.
    n_files : int, optional
        The total number of files that should be written. By default, a single
        file is output.
    use_dask : bool, optional
        Whether to use Dask for saving the data to the disk. May be relevant if
        a single frame cannot fit in memory.
    mode : str
        File opening mode, passed to h5py.

    Returns
    -------
    str, list of str
        Paths to the created files

    Raises
    ------
    AttributeError
        If the number of files exceeds the number of frames
    """

    if n_frames < n_files:
        msg = (
            "The number of written files must be less than the total "
            "number of frames (at least one frame per file)."
        )
        raise AttributeError(msg)

    # Generate coordinates associated with each dimension
    frame_coords = {
        "coord" + str(icoord): np.arange(s) for icoord, s in enumerate(frame_shape)
    }
    time_coord = np.array_split(np.arange(n_frames), n_files)
    filenames = [base_filename + str(f) + ".xrt" for f in np.arange(n_files)]

    for filename, local_time in zip(filenames, time_coord):
        with h5py.File(filename, mode=mode) as hfile:
            hfile.create_dataset("time", data=local_time)
            for key, val in frame_coords.items():
                hfile.create_dataset(key, data=val)
            for time in local_time:
                if not use_dask:
                    data_frame = np.random.random(frame_shape)
                    hfile.create_dataset(str(time), data=data_frame)
                else:
                    data_frame = da.random.random(frame_shape)
                    chunks = tuple(v for lst in data_frame.chunks for v in lst)
                    dset = hfile.create_dataset(
                        str(time),
                        shape=data_frame.shape,
                        chunks=chunks,
                        dtype="f8",
                    )
                    da.store(data_frame, dset)

    return filenames


def _name_matches_metadata(name):
    return name.startswith("coord") or name == "time"


def _assert_same_or_none(value, other):
    if value is None:
        return other
    if value != other:
        msg = "Inputs do not match."
        raise AssertionError(msg)
    return value


def read_data(*filenames):
    """
    Reads a list of filenames and extracts dataset locations

    The provided files are opened, checked for compatibility, and parsed for
    their stored data. The coordinates associated to this data are returned as
    a dictionnary object. The data itself is sorted and returned as strings
    describing where the data can be loaded from.

    Parameters
    ----------
    *filenames : str
        A list of paths to read the data from.

    Returns
    -------
    coords : dict
        A dictionnary mapping dimension names with a coordinate array, in the
        same fashion as those used in xarray.
    data_list : list
        A list with the same length as the number of frames that were found.
        Each item contains the name of the data file in which the frame is
        stored, as well as a dataset name.
    """
    if not len(filenames) > 0:
        msg = "Pass at least one filename."
        raise AttributeError(msg)

    _frame_shape = None
    _times = np.array([])
    data_list = []
    for filename in filenames:
        with h5py.File(filename, "r") as hfile:
            frame_names = [dset for dset in hfile if not _name_matches_metadata(dset)]
            frame_names = natsorted(frame_names)  # make sure the keys are sorted
            for dset in frame_names:
                try:
                    _frame_shape = _assert_same_or_none(_frame_shape, hfile[dset].shape)
                except AssertionError:
                    msg = "File contains heterogeneously sized frames."
                    raise ValueError(msg)

                data_list.append((filename, dset))

            # Read coordinates, overwrite everytime so we only keep the ones
            # from the last filename
            coord_names = [dset for dset in hfile if dset not in frame_names]
            coords = {
                cname: hfile[cname][:] for cname in coord_names if cname != "time"
            }
            _times = np.append(_times, hfile["time"])

    coords["time"] = _times

    return coords, data_list
