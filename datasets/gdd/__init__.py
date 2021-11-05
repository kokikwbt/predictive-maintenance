import datetime
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import seaborn as sns


def load_data(index='state'):

    assert index in ['state', 'anomaly', 'normal', 'linear', 'pressure']
    fp = os.path.dirname(__file__)

    if index == 'state':
        df = pd.read_csv(fp + '/Genesis_StateMachineLabel.csv.gz')
    elif index == 'anomaly':
        df = pd.read_csv(fp + '/Genesis_AnomalyLabels.csv.gz')
    elif index == 'normal':
        df = pd.read_csv(fp + '/Genesis_normal.csv.gz')
        df.Timestamp = df.Timestamp / 1000
    elif index == 'linear':
        df = pd.read_csv(fp + '/Genesis_lineardrive.csv.gz')
        df.Timestamp = df.Timestamp / 1000
    elif index == 'pressure':
        df = pd.read_csv(fp + '/Genesis_pressure.csv.gz')
        df.Timestamp = df.Timestamp / 1000

    df.Timestamp = df.Timestamp.apply(datetime.datetime.fromtimestamp)
    
    return df


def plot_genesis_labels(df, figsize=(15, 20), cmap='tab10'):
    """ Call this for machine states and anomaly labels """

    fig, ax = plt.subplots(10, figsize=figsize)

    df['MotorData.ActCurrent'].plot(ax=ax[0], legend=True, cmap=cmap)
    df['MotorData.ActPosition'].plot(ax=ax[1], legend=True, cmap=cmap)
    df['MotorData.ActSpeed'].plot(ax=ax[2], legend=True, cmap=cmap)

    df['MotorData.IsAcceleration'].plot(ax=ax[3], legend=True, cmap=cmap)
    df['MotorData.IsForce'].plot(ax=ax[4], legend=True, cmap=cmap)

    df[['MotorData.Motor_Pos1reached',    # binary
        'MotorData.Motor_Pos2reached',    # binary
        'MotorData.Motor_Pos3reached',    # binary
        'MotorData.Motor_Pos4reached',    # binary
    ]].plot(ax=ax[5], legend=True, cmap=cmap)

    df[['NVL_Recv_Ind.GL_Metall',  # binary
        'NVL_Recv_Ind.GL_NonMetall',  # binary
    ]].plot(ax=ax[6], legend=True, cmap=cmap)

    df[['NVL_Recv_Storage.GL_I_ProcessStarted',  # binary
        'NVL_Recv_Storage.GL_I_Slider_IN',  # binary
        'NVL_Recv_Storage.GL_I_Slider_OUT',  # binary
        'NVL_Recv_Storage.GL_LightBarrier',  # binary
        'NVL_Send_Storage.ActivateStorage',  # binary
    ]].plot(ax=ax[7], legend=True, cmap=cmap)

    df[['PLC_PRG.Gripper',  # binary
        'PLC_PRG.MaterialIsMetal',  # binary
    ]].plot(ax=ax[8], legend=True, cmap=cmap)

    df['Label'].plot(ax=ax[9], legend=True, cmap=cmap)

    for axi in ax:
        axi.set_xlim(0, df.shape[0])
        axi.set_ylabel('Value')

    ax[0].set_title('Date: {} to {}'.format(
        df.Timestamp.min(), df.Timestamp.max()))
    ax[-1].set_xlabel('Time')
    fig.tight_layout()

    return fig, ax
    
    
def plot_genesis_nonlabels(df, figsize=(15, 20), cmap='tab10'):
    """ Call this for non-labeled data """

    fig, ax = plt.subplots(8, figsize=figsize)

    df[['MotorData.SetCurrent',
        'MotorData.ActCurrent',
    ]].plot(ax=ax[0], legend=True, cmap=cmap)

    df[['MotorData.SetSpeed',
        'MotorData.ActSpeed',
    ]].plot(ax=ax[1], legend=True, cmap=cmap)

    df[['MotorData.SetAcceleration',
        'MotorData.IsAcceleration',
    ]].plot(ax=ax[2], legend=True, cmap=cmap)

    df[['MotorData.SetForce',
        'MotorData.IsForce'
    ]].plot(ax=ax[3], legend=True, cmap=cmap)

    df[['MotorData.Motor_Pos1reached',    # binary
        'MotorData.Motor_Pos2reached',    # binary
        'MotorData.Motor_Pos3reached',    # binary
        'MotorData.Motor_Pos4reached',    # binary
    ]].plot(ax=ax[4], legend=True, cmap=cmap)

    df[['NVL_Recv_Ind.GL_Metall',  # binary
        'NVL_Recv_Ind.GL_NonMetall',  # binary
    ]].plot(ax=ax[5], legend=True, cmap=cmap)

    df[['NVL_Recv_Storage.GL_I_ProcessStarted',  # binary
        'NVL_Recv_Storage.GL_I_Slider_IN',  # binary
        'NVL_Recv_Storage.GL_I_Slider_OUT',  # binary
        'NVL_Recv_Storage.GL_LightBarrier',  # binary
        'NVL_Send_Storage.ActivateStorage',  # binary
    ]].plot(ax=ax[6], legend=True, cmap=cmap)

    df[['PLC_PRG.Gripper',  # binary
        'PLC_PRG.MaterialIsMetal',  # binary
    ]].plot(ax=ax[7], legend=True, cmap=cmap)

    for axi in ax:
        axi.set_xlim(0, df.shape[0])
        axi.set_ylabel('Value')

    ax[0].set_title('Date: {} to {}'.format(df.Timestamp.min(), df.Timestamp.max()))
    ax[-1].set_xlabel('Time')

    fig.tight_layout()
    return fig, ax


def gen_summary(outdir=None):

    if outdir is None:
        outdir = os.path.dirname(__file__)

    os.makedirs(outdir, exist_ok=True)
    sns.set(font_scale=1.1, style='whitegrid')

    with PdfPages(outdir + '/gdd_summary.pdf') as pp:
        
        print('Plotting Genesis_StateMachineLabel...')
        df = load_data(index='state')
        fig, _ = plot_genesis_labels(df)
        fig.savefig(pp, bbox_inches='tight', format='pdf')
        plt.clf()
        plt.close()

        print('Plotting Genesis_AnomalyLabels...')
        df = load_data(index='anomaly')
        fig, _ = plot_genesis_labels(df)
        fig.savefig(pp, bbox_inches='tight', format='pdf')
        plt.clf()
        plt.close()

        print('Plotting Genesis_normal...')
        df = load_data(index='normal')
        fig, _ = plot_genesis_nonlabels(df)
        fig.savefig(pp, bbox_inches='tight', format='pdf')
        plt.clf()
        plt.close()

        print('Plotting Genesis_lineardrive...')
        df = load_data(index='linear')
        fig, _ = plot_genesis_nonlabels(df)
        fig.savefig(pp, bbox_inches='tight', format='pdf')
        plt.clf()
        plt.close()

        print('Plotting Genesis_pressure...')
        df = load_data(index='pressure')
        fig, _ = plot_genesis_nonlabels(df)
        fig.savefig(pp, bbox_inches='tight', format='pdf')
        plt.clf()
        plt.close()

        print("done!")