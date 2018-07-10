import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from collections import Counter


# Set plot preference which looks good to me.
def set_style(style='white', palette='deep', context='talk', font='Helvetica Neue', font_scale=1.25, rcparams={'figure.figsize': (11.7, 8.27)}):
    sns.set(style=style,
            palette=palette,
            context=context,
            font=font,
            font_scale=font_scale,
            rc=rcparams)


def show_or_save(file, dpi=150):
    """If file is specified, save the figure to the file with given resolution in dpi.
    Otherwise, show the figure.
    """
    plt.show() if file is None else plt.savefig(file, dpi=dpi)
    plt.clf()


def get_ax_to_draw(ax):
    """If ax is not specified, return an axis to draw a plot.
    Otherwise, return ax.
    """
    if ax:
        return ax
    else:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        return ax


def frequency(data, order=None, title=None, sort_by_values=False, ax=None, file=None, **kwargs):
    """Plot frequency bar chart.

    Examples:
        frequency([1, 2, 2, 3, 3, 3], order=[3, 1, 2], sort_by_values=True)

    Attributes:
        data (list): A list of elements.
        order (list): A list of elements which represents the order of the elements to be plotted.
        sort_by_values (bool): If True, the plot will be sorted in decreasing order of frequency values.
    """
    counter = Counter(data)
    if order is None:
        if sort_by_values:
            order = sorted(counter, key=counter.get, reverse=True)
        else:
            order = sorted(counter.keys())
    else:
        assert set(order) == set(counter.keys()), 'The order must contain all the elements.'
    counts = [counter[key] for key in order]

    ax = get_ax_to_draw(ax)

    # Some parameters used for plot configuration.
    height = max(counts) * 1.167
    xticks = list(range(len(order)))

    # Preset plot.
    ax.set_xticks(xticks)
    ax.set_xticklabels(order)
    ax.set_xlim([-0.66, len(counter) - 0.33])
    ax.set_ylim([0, height])
    if title is not None:
        ax.set_title(title)

    # Plot bar chart.
    ax.bar(x=xticks,
           height=counts,
           width=0.66,
           **kwargs)

    # Add text indicating frequency for each bar.
    for x, count in zip(xticks, counts):
        ax.text(x=x,
                y=count,
                s=str(count),
                size='small',
                va='bottom',
                ha='center')

    show_or_save(file)


def bar(data, keys=None, title=None, ordinal=True, horizontal=False, sort_by_values=False, file=None, **kwargs):
    if ordinal:
        if not horizontal:
            ordinal_vertical_bar_chart(data, keys=keys, title=title, file=file, sort_by_values=sort_by_values, **kwargs)
        else:
            ordinal_horizontal_bar_chart(data, keys=keys, title=title, file=file, sort_by_values=sort_by_values, **kwargs)

    else:
        raise NotImplementedError


def ordinal_vertical_bar_chart(data, keys=None, title=None, sort_by_values=False, axis=None, there_are_more_plots=False, file=None, **kwargs):
    counter = Counter(data)
    if keys is None:
        if sort_by_values:
            keys = sorted(counter, key=counter.get, reverse=True)
        else:
            keys = sorted(counter.keys())
    counts = [counter[key] for key in keys]

    # Some parameters used in plot configuration.
    height = max(counts) * 1.167
    xs = list(range(1, len(keys) + 1))

    # Preset plot.
    plt.xticks(xs, keys)
    plt.xlim([min(keys) - 0.66, max(keys) + 0.66])
    plt.ylim([0, height])
    if title is not None:
        plt.title(title)

    # Plot bar chart.
    plt.bar(x=xs,
            height=counts,
            width=0.66,
            **kwargs)

    # Add text indicating count value for each bar.
    for x, count in zip(xs, counts):
        plt.text(x=x,
                 y=count,
                 s=str(count),
                 size='small',
                 verticalalignment='bottom',
                 horizontalalignment='center')

    if not there_are_more_plots:
        show_or_save(file)


def ordinal_horizontal_bar_chart(data, keys=None, title=None, sort_by_values=False, there_are_more_plots=False, file=None, **kwargs):
        counter = Counter(data)
        if keys is None:
            if sort_by_values:
                keys = sorted(counter, key=counter.get)
            else:
                keys = sorted(counter.keys(), reverse=True)
        else:
            keys = keys[::-1]

        counts = [counter[key] for key in keys]

        # Some parameters used in plot configuration.
        width = max(counts) * 1.167
        ys = list(range(1, len(keys) + 1))

        # Preset plot.
        plt.xlim([0, width])
        plt.ylim([min(keys) - 0.66, max(keys) + 0.66])
        yticks = plt.yticks(ys, keys)
        if title is not None:
            plt.title(title)

        # Plot bar chart.
        plt.barh(y=ys,
                width=counts,
                height=0.66,
                **kwargs)

        # Add text indicating count value for each bar.
        for y, count in zip(ys, counts):
            plt.text(x=count + 0.0167 * width,
                     y=y,
                     s=str(count),
                     size='small',
                     verticalalignment='center',
                     horizontalalignment='left')

        if not there_are_more_plots:
            show_or_save(file)


def histogram(data, title=None, xlim=None, there_are_more_plots=False, file=None, **kwargs):

    if title is not None:
        plt.title(title)

    if xlim is not None:
        plt.xlim(xlim)

    plt.hist(data, color='black', ec='white', lw=1.33, **kwargs)
    if not there_are_more_plots:
        show_or_save(file)


def pca(data, labels=None, title=None, there_are_more_plots=False, file=None):
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

    if not there_are_more_plots:
        show_or_save(file)
