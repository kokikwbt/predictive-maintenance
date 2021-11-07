import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import tqdm
import pandas as pd


def parse_feature_names(fn):

    with open(fn) as f:
        names, lines = [], f.readlines()
        for line in lines:
            names.append(line.split('-')[-1].lstrip().rstrip())

    return names


def load_data(shorten_feature_names=True):

    fp = os.path.dirname(__file__)
    raw_data = np.loadtxt(fp + '/data.txt.gz')
    features = parse_feature_names(fp + '/Features.txt')

    if shorten_feature_names == True:
        for i in range(len(features) - 2):
            features[i] = features[i].split('(')[-1].split(')')[0].upper()

        features[-2] = 'comp_decay_state'
        features[-1] = 'turb_decay_state'

    return pd.DataFrame(raw_data, columns=features)


def normalize(df):
    df_norm = df.copy()

    for i in range(df_norm.shape[1]):
        if df_norm.iloc[:, i].max() - df_norm.iloc[:, i].min() > 0:
            maxv = df_norm.iloc[:, i].max()
            minv = df_norm.iloc[:, i].min()
            df_norm.iloc[:, i] = (df_norm.iloc[:, i] - minv) / (maxv - minv)
        else:
            df_norm.iloc[:, i] = 0.

    return df_norm


def load_clean_data():
    return normalize(load_data())


def gen_summary(wd=400, outdir=None):

    if outdir is None:
        outdir = os.path.dirname(__file__)

    os.makedirs(outdir, exist_ok=True)
    data = normalize(load_data(shorten_feature_names=False))
    
    with PdfPages(outdir + '/cbm_summary.pdf') as pp:
        for st in tqdm.trange(0, data.shape[0], wd):
            ed = st + wd
            fig, ax = plt.subplots(9, figsize=(24, 15))

            data.iloc[st:ed, 1].plot(legend=True, ax=ax[0])
            data.iloc[st:ed, 2].plot(legend=True, ax=ax[1])
            data.iloc[st:ed, 3:5].plot(legend=True, ax=ax[2])
            data.iloc[st:ed, 5:7].plot(legend=True, ax=ax[3])
            data.iloc[st:ed, 7:10].plot(legend=True, ax=ax[4])
            data.iloc[st:ed, 10:14].plot(legend=True, ax=ax[5])
            data.iloc[st:ed, 14].plot(legend=True, ax=ax[6])
            data.iloc[st:ed, 15].plot(legend=True, ax=ax[7])
            data.iloc[st:ed, [0, -2, -1]].plot(legend=True, ax=ax[8])

            ax[0].set_title('Normalized Sensor/Label data')
            ax[-1].set_xlabel('Time')
            for axi in ax: axi.set_ylabel('Value')

            fig.savefig(pp, bbox_inches='tight', format='pdf')
            plt.clf()
            plt.close()
    