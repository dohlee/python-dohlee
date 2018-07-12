import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sys

from adjustText import adjust_text
from sklearn.decomposition import PCA
from matplotlib.lines import Line2D
from collections import Counter
from functools import wraps

def get_ax_to_draw(ax, figsize=None):
    """If ax is not specified, return an axis to draw a plot.
    Otherwise, return ax.
    """
    if ax:
        return ax
    else:
        fig = plt.figure(figsize=figsize) if figsize else plt.figure()
        ax = fig.add_subplot(111)
        return ax


def try_save(file, dpi=150):
    """If file is specified, save the figure to the file with given resolution in dpi.
    Otherwise, show the figure.
    """
    return None if file is None else plt.savefig(file, dpi=dpi)


def save(file, dpi=120, tight_layout=True):
    """Save plot to a file.

    Attributes:
        file (str): Path to the resulting image file.
        dpi (int, default=120): Resolution.
        tight_layout (bool, default=True): Whether to run plt.tight_layout() before saving the plot.
    """
    if tight_layout:
        plt.tight_layout()
    plt.savefig(file, dpi=dpi)


def my_plot(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        ax = get_ax_to_draw(kwargs.get('ax', None), kwargs.get('figsize', None))
        if 'ax' in kwargs:
            del kwargs['ax']
        if 'figsize' in kwargs:
            del kwargs['figsize']

        if 'title' in kwargs:
            ax.set_title(kwargs['title'])
            del kwargs['title']
        if 'xlim' in kwargs:
            ax.set_xlim(kwargs['xlim'])
            del kwargs['xlim']
        if 'ylim' in kwargs:
            ax.set_ylim(kwargs['ylim'])
            del kwargs['ylim']
        if 'xlabel' in kwargs:
            ax.set_xlabel(kwargs['xlabel'])
            del kwargs['xlabel']
        if 'ylabel' in kwargs:
            ax.set_ylabel(kwargs['ylabel'])
            del kwargs['ylabel']

        result = func(*args, ax=ax, **kwargs)
        try_save(kwargs.get('file', None))
    return wrapper


# Set plot preference which looks good to me.
def set_style(style='white', palette='deep', context='talk', font='Helvetica Neue', font_scale=1.25, rcparams={'figure.figsize': (11.7, 8.27)}):
    sns.set(style=style,
            palette=palette,
            context=context,
            font=font,
            font_scale=font_scale,
            rc=rcparams)


@my_plot
def frequency(data, order=None, sort_by_values=False, ax=None, **kwargs):
    """Plot frequency bar chart.

    Examples:
        frequency([1, 2, 2, 3, 3, 3], order=[3, 1, 2], sort_by_values=True)

    Attributes:
        data (list): A list of elements.
        order (list): A list of elements which represents the order of the elements to be plotted.
        sort_by_values (bool): If True, the plot will be sorted in decreasing order of frequency values.
        ax (pyplot axis): Axis to draw the plot.
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


@my_plot
def histogram(data, ax=None, **kwargs):
    plt.hist(data, color='black', ec='white', lw=1.33, **kwargs)


@my_plot
def volcano(data, x, y, padj, label, cutoff=0.05, sample1=None, sample2=None, ax=None):
    """Draw a volcano plot.

    Examples:
        volcano(data=data, x='log2FoldChange', y='pvalue', label='Gene_Symbol', cutoff=0.05, padj='padj', figsize=(10.8, 8.4))

    Attributes:
        data (pandas dataframe): A dataframe resulting from DEG-discovery tool.
        x (str): Column name denoting log2 fold change.
        y (str): Column name denoting p-value.
                 (Note that p-values will be log10-transformed, so they should not be transformed beforehand.)
        padj (str): Column name denoting adjusted p-value.
        label (str): Column name denoting gene identifier.
        cutoff (float, optional): Adjusted p-value cutoff value to report significant DEGs.
        sample1 (str, optional): First sample name.
        sample2 (str, optional): Second sample name.
        ax (pyplot axis): Axis to draw the plot.
    """
    # Set x and y extent.
    x_limit = max(-np.min(data[x].values), np.max(data[x].values))
    x_extent = [-x_limit * 1.22, x_limit * 1.22]
    ax.set_xlim(x_extent)
    ax.set_ylim([0, max(-np.log10(data[y].values)) * 1.1])

    data_not_significant = data[data[padj] >= cutoff]
    ax.scatter(data_not_significant[x].values, -np.log10(data_not_significant[y].values), color='grey', marker='.')

    data_significant = data[data.padj < cutoff]
    ax.scatter(data_significant[x].values, -np.log10(data_significant[y].values), color='red', marker='.')

    texts = [ax.text(row[x], -np.log10(row[y]), row[label], fontsize=12) for _, row in data_significant.iterrows()]
    adjust_text(texts)

    line = Line2D([0], [0], color='red', lw=2.33, label='Adjusted p < %g' % cutoff)
    plt.legend(handles=[line])

    if (not sample1 is None) and (not sample2 is None):
        ax.set_xlabel(r'$log_2$FC ($log_{2}\frac{%s}{%s}$)' % (sample1.replace(' ', '\ '), sample2.replace(' ', '\ ')))
    else:
        ax.set_xlabel(r'$log_2$FC')
    ax.set_ylabel(r'$log_{10}$(p-value)')


def pca(data, labels=None, ax=None, **kwargs):
    '''Draw a simple principle component analysis plot of the data.'''
    # Fit PCA and get pc's
    pca = PCA(n_components=2)
    pca.fit(data)
    pc = pca.transform(data)

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
