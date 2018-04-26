import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from collections import Counter

sns.set(font='Helvetica Neue', style='white', context='paper', font_scale=1.66)


def show_or_save(file):
    plt.show() if file is None else plt.savefig(file)
    plt.clf()


def bar_chart(data, ordinal=True, keys=None, title=None, file=None, **kwargs):
    counter = Counter(data)
    if keys is None:
        keys = sorted(counter.keys()) if ordinal else counter.keys()
    counts = [counter[key] for key in keys]

    # Some parameters used in plot configuration.
    height = max(counts) * 1.167

    # Preset plot.
    plt.ylim([0, height])
    if title is not None:
        plt.title(title)

    # Plot bar chart.
    plt.bar(x=keys,
            height=counts,
            width=0.66,
            **kwargs)

    # Add text indicating count value for each bar.
    for key, count in zip(keys, counts):
        plt.text(x=key,
                 y=count,
                 s=str(count),
                 size='small',
                 verticalalignment='bottom',
                 horizontalalignment='center')

    show_or_save(file)


def histogram(data, title=None, xlim=None, file=None, **kwargs):

    if title is not None:
        plt.title(title)

    if xlim is not None:
        plt.xlim(xlim)

    plt.hist(data, color='black', ec='white', lw=1.33, **kwargs)
    show_or_save(file)


def pca(data, labels=None, title=None, file=None):
    '''Draw a simple principle component analysis plot of the data.'''
    # Fit PCA and get pc's
    pca = PCA(n_components=2)
    pca.fit(data)
    pc = pca.transform(data)

    if title is not None:
        plt.title(title)

    if labels is None:
        plt.scatter(x=pc[:, 0], y=pc[:, 1])

    else:
        # If labels are attached, color them in different colors
        labels = np.array(labels)
        for label in set(labels):
            toDraw = (labels == label)  # only draw these points this time

            plt.scatter(x=pc[toDraw, 0], y=pc[toDraw, 1], label=label)
            plt.legend(loc='best')

    # show explained variance ratio in the plot axes
    explainedVarianceRatio = pca.explained_variance_ratio_
    plt.xlabel('PC1 ({:.2%})'.format(explainedVarianceRatio[0]))
    plt.ylabel('PC2 ({:.2%})'.format(explainedVarianceRatio[1]))

    show_or_save(file)
