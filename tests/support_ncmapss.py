"""Shared N-CMAPSS HDF5 fixtures for loader and task tests."""

from pathlib import Path

import h5py
import numpy as np


def write_tiny_fixture(path: Path) -> None:
    a_dev = np.array(
        [
            [1, 1, 1, 1],
            [1, 2, 1, 1],
            [2, 1, 1, 0],
        ],
        dtype=np.float64,
    )
    a_test = np.array([[3, 1, 1, 1]], dtype=np.float64)
    w = np.ones((a_dev.shape[0], 4), dtype=np.float64)
    w_test = np.ones((1, 4), dtype=np.float64)
    xs = np.arange(a_dev.shape[0] * 14, dtype=np.float64).reshape(
        a_dev.shape[0], 14
    )
    xs_test = np.zeros((1, 14), dtype=np.float64)
    y_dev = np.array([[2], [1], [0]], dtype=np.int64)
    y_test = np.array([[5]], dtype=np.int64)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("A_dev", data=a_dev)
        handle.create_dataset("A_test", data=a_test)
        handle.create_dataset("W_dev", data=w)
        handle.create_dataset("W_test", data=w_test)
        handle.create_dataset("X_s_dev", data=xs)
        handle.create_dataset("X_s_test", data=xs_test)
        handle.create_dataset("X_v_dev", data=xs)
        handle.create_dataset("X_v_test", data=xs_test)
        handle.create_dataset("T_dev", data=np.zeros((3, 10)))
        handle.create_dataset("T_test", data=np.zeros((1, 10)))
        handle.create_dataset("Y_dev", data=y_dev)
        handle.create_dataset("Y_test", data=y_test)
        handle.create_dataset(
            "A_var",
            data=np.array([b"unit", b"cycle", b"Fc", b"hs"]),
        )
        handle.create_dataset(
            "W_var",
            data=np.array([b"alt", b"Mach", b"TRA", b"T2"]),
        )
        handle.create_dataset(
            "X_s_var",
            data=np.array([f"s{i}".encode() for i in range(14)]),
        )
        handle.create_dataset(
            "X_v_var",
            data=np.array([f"v{i}".encode() for i in range(14)]),
        )
        handle.create_dataset(
            "T_var",
            data=np.array([f"t{i}".encode() for i in range(10)]),
        )


def write_rich_fixture(path: Path) -> None:
    """Four development units and two test units with varying sensors."""
    a_dev = []
    y_dev = []
    xs_dev = []
    for unit, length in ((1, 4), (2, 3), (3, 5), (4, 2)):
        for cycle in range(1, length + 1):
            a_dev.append([unit, cycle, 1, 1])
            y_dev.append([length - cycle])
            xs_dev.append([float(unit + cycle + index) for index in range(14)])
    a_test = []
    y_test = []
    xs_test = []
    for unit, length, end_rul in ((10, 3, 7), (11, 2, 4)):
        for cycle in range(1, length + 1):
            a_test.append([unit, cycle, 1, 1])
            y_test.append([end_rul + length - cycle])
            xs_test.append(
                [float(unit + cycle + index) for index in range(14)]
            )
    with h5py.File(path, "w") as handle:
        handle.create_dataset("A_dev", data=np.asarray(a_dev, dtype=np.float64))
        handle.create_dataset(
            "A_test", data=np.asarray(a_test, dtype=np.float64)
        )
        handle.create_dataset(
            "W_dev", data=np.ones((len(a_dev), 4), dtype=np.float64)
        )
        handle.create_dataset(
            "W_test", data=np.ones((len(a_test), 4), dtype=np.float64)
        )
        handle.create_dataset(
            "X_s_dev", data=np.asarray(xs_dev, dtype=np.float64)
        )
        handle.create_dataset(
            "X_s_test", data=np.asarray(xs_test, dtype=np.float64)
        )
        handle.create_dataset("X_v_dev", data=np.asarray(xs_dev))
        handle.create_dataset("X_v_test", data=np.asarray(xs_test))
        handle.create_dataset("T_dev", data=np.zeros((len(a_dev), 10)))
        handle.create_dataset("T_test", data=np.zeros((len(a_test), 10)))
        handle.create_dataset(
            "Y_dev", data=np.asarray(y_dev, dtype=np.int64)
        )
        handle.create_dataset(
            "Y_test", data=np.asarray(y_test, dtype=np.int64)
        )
        handle.create_dataset(
            "A_var", data=np.array([b"unit", b"cycle", b"Fc", b"hs"])
        )
        handle.create_dataset(
            "W_var", data=np.array([b"alt", b"Mach", b"TRA", b"T2"])
        )
        handle.create_dataset(
            "X_s_var",
            data=np.array([f"s{i}".encode() for i in range(14)]),
        )
        handle.create_dataset(
            "X_v_var",
            data=np.array([f"v{i}".encode() for i in range(14)]),
        )
        handle.create_dataset(
            "T_var", data=np.array([f"t{i}".encode() for i in range(10)])
        )
