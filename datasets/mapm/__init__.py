import os
import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


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
