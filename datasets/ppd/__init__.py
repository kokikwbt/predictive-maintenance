import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import tqdm


def load_data(index=0):
    """ 0: C7
        1: C8
        2: C9
        3: C11
        4: C13
        5: C14
        6: C15
        7: C16

        Note that C7 and C13 included a short break
        (for about 100 timestamps long)
        between the two procedure.

    """
    fp = os.path.dirname(__file__)
    if index == 0:
        df = pd.read_csv(fp + '/C7-1.csv.gz')
        df = pd.concat([df, pd.read_csv(fp + '/C7-2.csv.gz')])
        df = df.reset_index(drop=True)
        df.Timestamp = df.index.values
        return df
    elif index == 1:
        return pd.read_csv(fp + '/C8.csv.gz')
    elif index == 2:
        return pd.read_csv(fp + '/C9.csv.gz')
    elif index == 3:
        return pd.read_csv(fp + '/C11.csv.gz')
    elif index == 4:
        df = pd.read_csv(fp + '/C13-1.csv.gz')
        df = pd.concat([df, pd.read_csv(fp + '/C13-2.csv.gz')])
        df = df.reset_index(drop=True)
        df.Timestamp = df.index.values
        return df
    elif index == 5:
        return pd.read_csv(fp + '/C14.csv.gz')
    elif index == 6:
        return pd.read_csv(fp + '/C15.csv.gz')
    elif index == 7:
        return pd.read_csv(fp + '/C16.csv.gz')
    else:
        raise ValueError


def rename_components(df):
    """ current and speed
    """
    # Rename L
    L_curr = ['L_1', 'L_3', 'L_4', 'L_7', 'L_9']
    L_speed = ['L_2', 'L_6', 'L_5', 'L_8', 'L_10']
    df = df.rename(columns={k: f'c{i}_curr' for i, k in enumerate(L_curr)})
    df = df.rename(columns={k: f'c{i}_speed' for i, k in enumerate(L_speed)})

    # Rename A, B, and C
    df = df.rename(columns={f'A_{i}': f'c5_val{i}' for i in range(1, 6)})
    df = df.rename(columns={f'B_{i}': f'c6_val{i}' for i in range(1, 6)})
    df = df.rename(columns={f'C_{i}': f'c7_val{i}' for i in range(1, 6)})

    return df


def gen_summary(outdir=None):

    if outdir is None:
        outdir = os.path.dirname(__file__)

    os.makedirs(outdir, exist_ok=True)
    sns.set(font_scale=1.1, style='whitegrid')

    with PdfPages(outdir + '/ppd_summary.pdf') as pp:
        for i in tqdm.trange(8):
            df = load_data(index=i)
            df = rename_components(df)
            fig, ax = plt.subplots(8, figsize=(20, 20))

            for i in range(8):
                df.loc[:, df.columns.str.contains(f'c{i}')].plot(
                    ax=ax[i], legend=True, cmap='tab10')
                ax[i].set_ylabel("Value")
                ax[i].set_xlim(0, df.shape[0])

            ax[-1].set_xlabel('Time')
            fig.savefig(pp, bbox_inches='tight', format='pdf')
            plt.clf()
            plt.close()