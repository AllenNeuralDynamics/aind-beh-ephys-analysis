import h5py
import numpy as np
from pathlib import Path
import probeinterface as pi

import spikeinterface as si
from spikeinterface import BaseRecording, BaseRecordingSegment
__version__ = '0.1.0'


# class TransposedDataset:
#     def __init__(self, dataset):
#         self.dataset = dataset
#         self.shape = (dataset.shape[1], dataset.shape[0])  # (channels, time)
#         self.dtype = dataset.dtype

#     def __getitem__(self, idx):
#         # If idx is a slice (e.g. [0:10]), we want time indices
#         if isinstance(idx, (slice, int)):
#             data = self.dataset[:, idx]  # shape: (time, channels)
#             return data.T  # shape: (channels, time)

#         # If idx is a tuple (row, col) in the transposed view
#         elif isinstance(idx, tuple) and len(idx) == 2:
#             chan_idx, time_idx = idx
#             return self.dataset[time_idx, chan_idx]  # swap indices

#         else:
#             raise IndexError("Unsupported index type for TransposedDataset")

#     # Optional: implement __len__ or other methods if needed
#     def __len__(self):
#         return self.shape[0]

class HDF5Recording(BaseRecording):
    """
    SpikeInterface recording extractor for HDF5 format neural data.

    Loads neural recording data stored in HDF5 format and provides
    a SpikeInterface-compatible interface.
    """

    def __init__(self, file_path: Path | str):
        """
        Initialize HDF5 recording extractor.

        Parameters
        ----------
        file_path : Path or str
            Path to HDF5 file containing neural recording data
        """
        self._h5file = h5py.File(file_path, mode="r")

        sampling_frequency = self._h5file.attrs["SamplingFrequency"]

        samples = self._h5file["/samples"]
        print(samples.shape)
        # samples = TransposedDataset(samples_ds)  # lazy transposed wrapper
        # samples = samples_ds

        timestamps = self._h5file["/timestamps"][:, 0]
        num_channels = samples.shape[0]
        channel_ids = [str(ch) for ch in range(num_channels)]
        dtype = samples.dtype

        BaseRecording.__init__(self, sampling_frequency, channel_ids, dtype)

        rec_segment = HDF5RecordingSegment(samples, time_vector=timestamps)
        self.add_recording_segment(rec_segment)

        # make a tetrode probe
        probegroup = pi.ProbeGroup()
        for i in range(num_channels // 4):
            tetrode = pi.generate_tetrode()
            tetrode.move([i * 50, 0])
            probegroup.add_probe(tetrode)
        probegroup.set_global_device_channel_indices(np.arange(num_channels))
        self.set_probegroup(probegroup, in_place=True)

        gain = self._h5file.attrs["ADBitVolts"][0] * 1e6
        self.set_channel_gains([gain] * self.get_num_channels())
        self.set_channel_offsets([0] * self.get_num_channels())

        self._kwargs = {"file_path": str(Path(file_path).absolute())}

        


class HDF5RecordingSegment(BaseRecordingSegment):
    """
    Recording segment class for HDF5-backed neural data.
    """

    def __init__(self, h5_dataset, **time_kwargs):
        """
        Initialize recording segment.

        Parameters
        ----------
        h5_dataset : h5py.Dataset
            HDF5 dataset containing timeseries data
        **time_kwargs : dict
            Additional timing arguments passed to BaseRecordingSegment
        """
        BaseRecordingSegment.__init__(self, **time_kwargs)
        self._timeseries = h5_dataset

    def get_num_samples(self) -> int:
        """
        Get the number of samples in this signal block.

        Returns
        -------
        int
            Number of samples in the signal block
        """
        return self._timeseries.shape[1]

    def get_traces(
        self,
        start_frame: int | None = None,
        end_frame: int | None = None,
        channel_indices: list[int | str] | None = None,
    ) -> np.ndarray:
        """
        Extract neural traces for specified time range and channels.

        Parameters
        ----------
        start_frame : int, optional
            Starting frame index (default: None, uses beginning)
        end_frame : int, optional
            Ending frame index (default: None, uses end)
        channel_indices : list of int or str, optional
            Channel indices to extract (default: None, uses all channels)

        Returns
        -------
        np.ndarray
            Extracted traces with shape (n_samples, n_channels)
        """
        traces = self._timeseries[:, start_frame:end_frame].T
        if channel_indices is not None:
            traces = traces[:, channel_indices]
        return traces
    
    # def get_start_time(self) -> float:
    #     if self._time_vector is not None:
    #         return float(self._time_vector[0])
    #     return 0.0

    # def get_end_time(self) -> float:
    #     if self._time_vector is not None:
    #         return float(self._time_vector[-1])
    #     return float(self.get_num_samples())


read_hdf5 = HDF5Recording