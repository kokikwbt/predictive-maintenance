import os
import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from collections import Counter


def load_data():
    fp = os.path.dirname(__file__)

    # Sensor data
    data = pd.read_csv(fp + '/PdM_telemetry.csv.gz')

    # Error alarm logs
    data = data.merge(
        pd.read_csv(fp + '/PdM_errors.csv.gz'),
        how='left', on=['datetime', 'machineID'])

    # Failure logs
    data = data.merge(
        pd.read_csv(fp + '/PdM_failures.csv.gz'),
        how='left', on=['datetime', 'machineID'])

    # Formatting
    data.datetime = pd.to_datetime(data.datetime)

    return data


def cleaning(df):

    # NaN values are encoded to -1
    df = df.sort_values('errorID')
    df.errorID = df.errorID.factorize()[0]
    df = df.sort_values('failure')
    df.failure = df.failure.factorize()[0]
    df = df.sort_values(['machineID', 'datetime'])

    df.errorID = df.errorID.astype('category')
    df.failure = df.failure.astype('category')

    df.volt = df.volt.astype('float32')
    df.rotate = df.rotate.astype('float32')
    df.pressure = df.pressure.astype('float32')
    df.vibration = df.vibration.astype('float32')

    df.datetime = pd.to_datetime(df.datetime)
    return df


def load_clean_data():
    return cleaning(load_data())


def generate_run_to_failure(raw_data, health_cencor_aug=1000,
                            min_lifetime=2, max_lifetime=300,
                            seed=123, outfn=None):

    run_to_failure = []
    error_ids = raw_data.errorID.dropna().sort_values().unique().tolist()

    for machine_id, g in tqdm.tqdm(raw_data.groupby('machineID'), desc='run-to-failure'):
        g = g.set_index('datetime').sort_index()

        start_date = g.index.values[0]
        failures = g.loc[~g.failure.isnull()]

        for event_time, event in failures.iterrows():
            # Extracting a single cycle/process
            cycle = g[start_date:event_time].drop('machineID', axis=1)

            lifetime = (event_time - start_date).days
            if lifetime < 1:
                start_date = event_time
                continue

            numerical_features = cycle.agg(['min', 'max', 'mean']).unstack().reset_index()
            numerical_features['feature'] = numerical_features.level_0.str.cat(numerical_features.level_1, sep='_')
            numerical_features = numerical_features.pivot_table(columns='feature', values=0)

            categorical_features = pd.DataFrame(Counter(cycle.errorID), columns=error_ids, index=[0])

            sample = pd.concat([numerical_features, categorical_features], axis=1)
            sample[['machine_id', 'lifetime', 'broken']] = machine_id, lifetime, 1

            run_to_failure.append(sample)
            start_date = event_time

    run_to_failure = pd.concat(run_to_failure, axis=0).fillna(0).reset_index(drop=True)

    health_censors = cencoring_augmentation(raw_data,
        n_samples=health_cencor_aug,
        min_lifetime=min_lifetime,
        max_lifetime=max_lifetime,
        seed=seed)

    run_to_failure = pd.concat([run_to_failure, health_censors])

    # Shuffle
    run_to_failure = run_to_failure.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    if outfn is not None:
        run_to_failure.to_csv(outfn, index=False)

    return run_to_failure


def cencoring_augmentation(raw_data, n_samples=10, max_lifetime=150, min_lifetime=2, seed=123):

    error_ids = raw_data.errorID.dropna().sort_values().unique().tolist()
    np.random.seed(seed)
    samples = []
    pbar = tqdm.tqdm(total=n_samples, desc='augmentation')

    while len(samples) < n_samples:
        
        cencor_timing = np.random.randint(min_lifetime, max_lifetime)
        machine_id = np.random.randint(100) + 1
        tmp = raw_data[raw_data.machineID == machine_id]
        tmp = tmp.drop('machineID', axis=1).set_index('datetime').sort_index()

        failures = tmp[~tmp.failure.isnull()]
        if failures.shape[0] < 2:
            continue

        failure_id = np.random.randint(failures.shape[0])
        failure = failures.iloc[failure_id]
        event_time = failure.name
        start_date = tmp.index.values[0] if failure_id == 0 else failures.iloc[failure_id - 1].name

        # cencoring
        cycle = tmp[start_date:event_time]
        cycle = cycle.iloc[:cencor_timing]

        if not cycle.shape[0] == cencor_timing:
            continue

        numerical_features = cycle.agg(['min', 'max', 'mean']).unstack().reset_index()
        numerical_features['feature'] = numerical_features.level_0.str.cat(numerical_features.level_1, sep='_')
        numerical_features = numerical_features.pivot_table(columns='feature', values=0)

        categorical_features = pd.DataFrame(Counter(cycle.errorID), columns=error_ids, index=[0])

        sample = pd.concat([numerical_features, categorical_features], axis=1)
        sample[['machine_id', 'lifetime', 'broken']] = machine_id, cencor_timing, 0
        samples.append(sample)
        pbar.update(1)

    pbar.close()
    return pd.concat(samples).reset_index(drop=True).fillna(0)


def plot_sequence_and_events(data, machine_id=1):

    data = data[data.machineID == machine_id]
    fig, ax = plt.subplots(4 + 2, figsize=(8, 8))

    data.plot(y='volt', legend=True, ax=ax[0])
    data.plot(y='rotate', legend=True, ax=ax[1])
    data.plot(y='pressure', legend=True, ax=ax[2])
    data.plot(y='vibration', legend=True, ax=ax[3])

    if data.errorID.isnull().sum() < data.errorID.shape[0]:
        pd.get_dummies(data.errorID).plot(ax=ax[4])
    if data.failure.isnull().sum() < data.failure.shape[0]:
        pd.get_dummies(data.failure).plot(ax=ax[5])

    ax[0].set_title('Machine #{}'.format(machine_id))

    for i in range(5):
        ax[i].set_xlabel(None)
        ax[i].set_xticklabels([])

    fig.tight_layout()

    return fig, ax


def gen_summary(outdir=None):

    if outdir is None:
        outdir = os.path.dirname(__file__)

    os.makedirs(outdir, exist_ok=True)
    df = load_data()

    with PdfPages(outdir + '/mapm_summary.pdf') as pp:
        for i in tqdm.trange(1, 101):
            fig, _ = plot_sequence_and_events(df, machine_id=i)
            fig.savefig(pp, format='pdf')
            plt.clf()
            plt.close()
