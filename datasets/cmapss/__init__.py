import os
import numpy as np
import pandas as pd


def load_data(index="FD004"):
    assert index in ["FD001", "FD002", "FD003", "FD004"]
    filepath = os.path.dirname(__file__)

    print("-----------------")
    print(f" Data Set: {index} ")
    print("-----------------")

    if index == "FD001":
        print("Train trjectories: 100")
        print("Test trajectories: 100")
        print("Conditions: ONE (Sea Level)")
        print("Fault Modes: ONE (HPC Degradation)\n")

    if index == "FD002":
        print("Train trjectories: 260")
        print("Test trajectories: 259")
        print("Conditions: SIX")
        print("Fault Modes: ONE (HPC Degradation)\n")

    if index == "FD003":
        print("Train trjectories: 100")
        print("Test trajectories: 100")
        print("Conditions: ONE (Sea Level)")
        print("Fault Modes: TWO (HPC Degradation, Fan Degradation)\n")

    if index == "FD004":
        print("Train trjectories: 248")
        print("Test trajectories: 249")
        print("Conditions: SIX")
        print("Fault Modes: TWO (HPC Degradation, Fan Degradation)\n")

    # Original data
    train_set = np.loadtxt(filepath + f"/train_{index}.txt.gz")
    test_set  = np.loadtxt(filepath + f"/test_{index}.txt.gz")
    labels    = np.loadtxt(filepath + f"/RUL_{index}.txt.gz")  # RUL: Remaining Useful Life

    # Convert to pandas.DataFrame objects
    col_names = ["unit number", "time"]
    col_names += [f"operational setting {i}" for i in range(1, 4)]
    col_names += [f"sensor measurement {i}" for i in range(1, 22)]
    train_set = pd.DataFrame(train_set, columns=col_names)
    test_set  = pd.DataFrame(test_set, columns=col_names)

    def set_dtype(df):
        return df.astype({"unit number": np.int64, "time": np.int64})

    train_set = set_dtype(train_set)
    test_set  = set_dtype(test_set)

    return train_set, test_set, labels


def cleaning(df):
    raise NotImplementedError


def load_clean_data():
    raise NotImplementedError