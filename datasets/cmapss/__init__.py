import os
import numpy as np
import pandas as pd
import tqdm


_applied_features = [2, 3, 4, 7, 11, 12, 15]


def load_data(index="FD004", features=None):

    if type(index) == str:
        assert index in ["FD001", "FD002", "FD003", "FD004"]
    elif type(index) == int:
        assert index in [0, 1, 2, 3]
        index = f'FD00{index+1}'

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
        print("Train trjectories: 249")
        print("Test trajectories: 248")
        print("Conditions: SIX")
        print("Fault Modes: TWO (HPC Degradation, Fan Degradation)\n")

    # Original data
    train_set = np.loadtxt(filepath + f"/train_{index}.txt.gz")
    test_set  = np.loadtxt(filepath + f"/test_{index}.txt.gz")
    labels    = np.loadtxt(filepath + f"/RUL_{index}.txt.gz")  # RUL: Remaining Useful Life

    # Convert to pandas.DataFrame objects
    col_names = ["unit_number", "time"]
    col_names += [f"operation{i}" for i in range(1, 4)]
    col_names += [f"sensor{i}" for i in range(1, 22)]
    train_set = pd.DataFrame(train_set, columns=col_names)
    test_set  = pd.DataFrame(test_set, columns=col_names)

    def set_dtype(df):
        return df.astype({"unit_number": np.int64, "time": np.int64})

    def extract_features(df, features):
        columns = ['unit_number', 'time']
        columns += [f'sensor{i}' for i in features]
        return df.loc[:, columns]

    train_set = set_dtype(train_set)
    test_set  = set_dtype(test_set)

    if features is not None:
        train_set = extract_features(train_set, features)
        test_set  = extract_features(test_set, features)

    return train_set, test_set, labels


def cleaning(df):
    raise NotImplementedError


def load_clean_data():
    raise NotImplementedError


def load_mesurement_list(
    index="FD004", 
    features=[2, 3, 4, 7, 11, 12 ,15],
    ):
    """
    * transform train_set and test_set into the lists of
        multivariate senser mesurements according to unit numbers.
    * features: the default features were applied in the previous research, 
        "A Similarity-Based Prognostics Approach
        for Remaining Useful Life Estimation of Engineered Systems".
    """
    assert index in ["FD001", "FD002", "FD003", "FD004"]
    train_set, test_set, labels = load_data(index=index)

    refined_train_set =[]
    for _, seq_df in train_set.groupby("unit_number"):
        seq_df  = seq_df.sort_values("time")
        ex_seq_df = seq_df[[f"sensor{f_id}" for f_id in features]].reset_index(drop=True)
        refined_train_set.append(ex_seq_df)

    refined_test_set =[]
    for _, seq_df in test_set.groupby("unit_number"):
        seq_df  = seq_df.sort_values("time")
        ex_seq_df = seq_df[[f"sensor{f_id}" for f_id in features]].reset_index(drop=True)
        refined_test_set.append(ex_seq_df)

    return refined_train_set, refined_test_set, labels


def run_to_failure_aux(df, lifetime, unit_number):

    assert lifetime <= df.shape[0]
    broken = 0 if lifetime < df.shape[0] else 1
    sample = pd.DataFrame(
        {'lifetime': lifetime, 'broken': broken, 'unit_number': unit_number}, index=[0])

    sensors = df.loc[:, df.columns.str.contains('sensor')]
    num_features = sensors.iloc[:lifetime].agg(['min', 'max', 'mean', 'std'])
    num_features = num_features.unstack().reset_index()
    num_features['feature'] = num_features.level_0.str.cat(
        num_features.level_1, sep='_')
    num_features = num_features.pivot_table(columns='feature', values=0)

    return pd.concat([sample, num_features], axis=1)


def censoring_augmentation(raw_data, n_samples=10, seed=123):

    np.random.seed(seed)
    datasets = [g for _, g in raw_data.groupby('unit_number')]
    timeseries = raw_data.groupby('unit_number').size()
    samples = []
    pbar = tqdm.tqdm(total=n_samples, desc='augmentation')

    while len(samples) < n_samples:
        # draw a machine
        unit_number = np.random.randint(timeseries.shape[0])
        censor_timing = np.random.randint(timeseries.iloc[unit_number])
        sample = run_to_failure_aux(
            datasets[unit_number], censor_timing, unit_number)
        samples.append(sample)
        pbar.update(1)

    return pd.concat(samples).reset_index(drop=True).fillna(0)


def generate_run_to_failure(df, health_censor_aug=0, seed=123):

    samples = []
    for unit_id, timeseries in tqdm.tqdm(df.groupby('unit_number'), desc='RUL'):
        samples.append(run_to_failure_aux(
            timeseries, timeseries.shape[0], unit_id))

    samples = pd.concat(samples)

    if health_censor_aug > 0:
        aug_samples = censoring_augmentation(
            df, n_samples=health_censor_aug, seed=seed)
        return pd.concat([samples, aug_samples]).reset_index(drop=True)
    else:
        return samples.reset_index(drop=True)


def leave_one_out(target='run-to-failure',
                  health_censor_aug=1000, seed=123,
                  input_fn=None, output_fn=None):

    if input_fn is not None:
        subsets = pd.read_csv(input_fn)

    else:
        subsets = []
        for index in range(4):
            raw_data = load_data(index=index)[0]
            raw_data = raw_data.assign(machine_id=index)

            if target == 'run-to-failure':
                subset = generate_run_to_failure(
                    raw_data, health_censor_aug, seed)
                subset = subset.assign(fold=index)
                subsets.append(subset)

            elif target == 'time-to-failure':
                raise NotImplementedError

            else:
                raise ValueError

        subsets = pd.concat(subsets).reset_index(drop=True)

    if output_fn is not None:
        subsets.to_csv(output_fn, index=False)

    # List of tuples: (train_data, test_data)
    train_test_sets = [(
        subsets[subsets.fold != i].reset_index(drop=True),
        subsets[subsets.fold == i].reset_index(drop=True)) for i in range(4)]

    return train_test_sets


def generate_validation_sets(method='leave-one-out', n_splits=5, seed=123, outdir=None):
    validation_sets = []

    if method == 'kfold':
        raise NotImplementedError

    elif method == 'leave-one-out':
        validation_sets = leave_one_out(target='run-to-failure',
                                        health_censor_aug=1000,
                                        seed=seed)

        if outdir is not None:
            for i, (train_data, test_data) in enumerate(validation_sets):
                train_data.to_csv(outdir + f'/train_{i}.csv.gz', index=False)
                test_data.to_csv(outdir + f'/test_{i}.csv.gz', index=False)        

    return validation_sets


def load_validation_sets(filepath, method='leave-one-out', n_splits=5):
    if method == 'kfold':
        return [(pd.read_csv(filepath + f'/train_{i}.csv.gz'),
                 pd.read_csv(filepath + f'/test_{i}.csv.gz'))
                 for i in range(n_splits)]

    elif method == 'leave-one-out':
        return [(pd.read_csv(filepath + f'/train_{i}.csv.gz'),
                 pd.read_csv(filepath + f'/test_{i}.csv.gz'))
                 for i in range(4)]
