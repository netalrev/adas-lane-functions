"""
src/features/dataset_writer.py
================================
HDF5 dataset accumulator and writer.

Public API
----------
    DatasetWriter(output_path)
        Instantiate with a path to the .h5 file to produce.

    DatasetWriter.add_sample(mf, cipv, lane_assignment, cut_in,
                              segment_name, track_id, frame_idx)
        Accumulate one training sample in memory.

    DatasetWriter.flush() -> int
        Persist all accumulated samples to disk.  Returns the total
        number of samples written.  Safe to call multiple times
        (each call appends to the existing file if it already exists).

HDF5 schema
-----------
    /mf_sequences   [N, T, D]  float32 — stacked MF windows
    /cipv_labels    [N]        int8    — 0 or 1
    /lane_labels    [N]        int8    — −2..+2
    /cut_in_labels  [N]        int8    — 0 or 1
    /segment_ids    [N]        int32   — index into /segment_names
    /track_ids      [N]        int32   — tracker track ID
    /frame_indices  [N]        int32   — frame index within the segment
    /segment_names  [S]        variable-length UTF-8 string

All per-sample arrays grow along axis 0 with resize().  The initial
chunking strategy uses chunk_size=256 samples for efficient sequential
reads during training.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import h5py


_CHUNK_SIZE = 256   # HDF5 chunk size (number of samples per chunk)


class DatasetWriter:
    """
    Accumulates training samples and writes them to an HDF5 file.

    Parameters
    ----------
    output_path : str | Path
        Destination HDF5 file.  The parent directory is created if needed.
        If the file already exists, new samples are appended.
    """

    def __init__(self, output_path: str | Path) -> None:
        self._path = Path(output_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

        # In-memory staging buffers
        self._mf:           list[np.ndarray] = []
        self._cipv:         list[int]        = []
        self._lane:         list[int]        = []
        self._cut_in:       list[int]        = []
        self._seg_names:    list[str]        = []
        self._seg_ids:      list[int]        = []
        self._track_ids:    list[int]        = []
        self._frame_idxs:   list[int]        = []

        # Map segment_name → integer segment ID (consistent across flush calls)
        self._seg_name_to_id: dict[str, int] = {}
        # Existing segment names already written to disk (loaded on first flush)
        self._disk_seg_names: Optional[list[str]] = None

    def add_sample(
        self,
        mf:              np.ndarray,
        cipv:            int,
        lane_assignment: int,
        cut_in:          int,
        segment_name:    str,
        track_id:        int,
        frame_idx:       int,
    ) -> None:
        """
        Stage one training sample.

        Parameters
        ----------
        mf : np.ndarray
            Shape [T, D] float32 — the MF window produced by MFAssembler.
        cipv : int
            CIPV label (0 or 1).
        lane_assignment : int
            Lane assignment label (−2..+2).
        cut_in : int
            Cut-In label (0 or 1).
        segment_name : str
            TFRecord segment name (used for /segment_names).
        track_id : int
            Tracker track ID.
        frame_idx : int
            Frame index within this segment.
        """
        if segment_name not in self._seg_name_to_id:
            self._seg_name_to_id[segment_name] = len(self._seg_name_to_id)
        seg_id = self._seg_name_to_id[segment_name]

        self._mf.append(mf.astype(np.float32))
        self._cipv.append(int(cipv))
        self._lane.append(int(lane_assignment))
        self._cut_in.append(int(cut_in))
        self._seg_names.append(segment_name)
        self._seg_ids.append(seg_id)
        self._track_ids.append(int(track_id))
        self._frame_idxs.append(int(frame_idx))

    def flush(self) -> int:
        """
        Write all staged samples to the HDF5 file and clear the buffers.

        Returns
        -------
        int
            Total number of samples now in the file after this flush.
        """
        if not self._mf:
            return 0

        n_new = len(self._mf)
        mf_arr   = np.stack(self._mf, axis=0)   # [N, T, D]
        T, D     = mf_arr.shape[1], mf_arr.shape[2]

        cipv_arr   = np.array(self._cipv,      dtype=np.int8)
        lane_arr   = np.array(self._lane,       dtype=np.int8)
        cut_in_arr = np.array(self._cut_in,     dtype=np.int8)
        seg_id_arr = np.array(self._seg_ids,    dtype=np.int32)
        trk_arr    = np.array(self._track_ids,  dtype=np.int32)
        frm_arr    = np.array(self._frame_idxs, dtype=np.int32)

        with h5py.File(self._path, "a") as f:
            total = self._append_dataset(
                f, "mf_sequences",  mf_arr,
                shape0=0, dtype=np.float32,
                chunk=(min(_CHUNK_SIZE, n_new), T, D),
                extra_dims=(T, D),
            )
            self._append_dataset(f, "cipv_labels",   cipv_arr,   dtype=np.int8)
            self._append_dataset(f, "lane_labels",   lane_arr,   dtype=np.int8)
            self._append_dataset(f, "cut_in_labels", cut_in_arr, dtype=np.int8)
            self._append_dataset(f, "segment_ids",   seg_id_arr, dtype=np.int32)
            self._append_dataset(f, "track_ids",     trk_arr,    dtype=np.int32)
            self._append_dataset(f, "frame_indices", frm_arr,    dtype=np.int32)

            # segment_names: variable-length UTF-8, append unique new names
            dt = h5py.special_dtype(vlen=str)
            if "segment_names" not in f:
                f.create_dataset("segment_names", data=np.array([], dtype=object),
                                 dtype=dt, maxshape=(None,),
                                 chunks=(max(_CHUNK_SIZE, 1),))
            existing_names = list(f["segment_names"])
            for name in self._seg_names:
                if name not in existing_names:
                    existing_names.append(name)
            f["segment_names"].resize((len(existing_names),))
            f["segment_names"][:] = existing_names

        # Clear staging buffers
        self._mf.clear()
        self._cipv.clear()
        self._lane.clear()
        self._cut_in.clear()
        self._seg_names.clear()
        self._seg_ids.clear()
        self._track_ids.clear()
        self._frame_idxs.clear()

        return int(total)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _append_dataset(
        f: h5py.File,
        name: str,
        data: np.ndarray,
        *,
        shape0: int = 0,
        dtype=None,
        chunk=None,
        extra_dims: tuple = (),
    ) -> int:
        """
        Create or resize-and-append a dataset in an open HDF5 file.

        Returns the new total number of rows (axis-0 length).
        """
        n = data.shape[0]
        if name not in f:
            if chunk is None:
                chunk = (min(_CHUNK_SIZE, n),) + extra_dims
            max_shape = (None,) + extra_dims
            full_shape = (n,) + extra_dims
            f.create_dataset(
                name, data=data, dtype=dtype,
                maxshape=max_shape, chunks=chunk,
            )
            return n
        else:
            dset = f[name]
            old_n = dset.shape[0]
            new_n = old_n + n
            dset.resize(new_n, axis=0)
            dset[old_n:] = data
            return new_n
