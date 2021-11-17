import matplotlib.pyplot as plt
from sklearn import model_selection


def train_test_split(arrays, test_size=0.2, random_state=None, shuffle=False):
    return model_selection.train_test_split(
        arrays=arrays,
        test_size=test_size,
        random_state=random_state,
        shuffle=shuffle)


def hist_survival_time(data, ax=None, figsize=(6, 4), bins_0=20, bins_1=30):

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    time_0 = data.loc[data['broken'] == 0, 'lifetime']
    ax.hist(time_0, bins=bins_0, alpha=0.3, color='blue', label='not broken yet')

    time_1 = data.loc[data['broken'] == 1, 'lifetime']
    ax.hist(time_1, bins=bins_1, alpha=0.7, color='black', label='broken')

    ax.set_title( 'Histogram - survival time', fontsize=15)

    if ax is None:
        return fig, ax
    else:
        return ax