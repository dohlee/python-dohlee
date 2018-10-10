import matplotlib
matplotlib.use('Agg')

import sys
import os
import itertools
import dohlee.analysis as analysis
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import scipy.stats as stats
import seaborn as sns

from adjustText import adjust_text
from collections import Counter
from sklearn.decomposition import PCA
from matplotlib.lines import Line2D
from collections import Counter
from functools import wraps
from fastTSNE import TSNE
from urllib import request
from umap import UMAP


def _get_ax_to_draw(ax, figsize=None):
    """If ax is not specified, return an axis to draw a plot.
    Otherwise, return ax.
    """
    if ax:
        return ax
    else:
        if figsize:
            return get_axis(figsize=figsize)
        else:
            return get_axis(preset='wide')


def _try_save(file, dpi=300):
    """If file is specified, save the figure to the file with given resolution in dpi.
    Otherwise, show the figure.
    """
    return None if file is None else plt.savefig(file, dpi=dpi)


def save(file, dpi=300, tight_layout=True):
    """Save plot to a file.

    :param str file: Path to the resulting image file.
    :param int dpi: (default=300) Resolution.
    :param bool tight_layout: (default=True) Whether to run plt.tight_layout() before saving the plot.
    """
    if tight_layout:
        plt.tight_layout()
    plt.savefig(file, dpi=dpi)


def clear():
    """Clear the plot.
    """
    plt.clf()
    plt.close()


def _my_plot(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        ax = _get_ax_to_draw(kwargs.get('ax', None), kwargs.get('figsize', None))
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

        file_path = kwargs.get('file', None)
        if 'file' in kwargs:
            del kwargs['file']

        rotate_xticklabels = kwargs.get('rotate_xticklabels', 0)
        if 'rotate_xticklabels' in kwargs:
            del kwargs['rotate_xticklabels']

        xticklabels = kwargs.get('xticklabels', True)
        if 'xticklabels' in kwargs:
            del kwargs['xticklabels']

        yticklabels = kwargs.get('yticklabels', True)
        if 'yticklabels' in kwargs:
            del kwargs['yticklabels']

        legend_size = kwargs.get('legend_size', None)
        if 'legend_size' in kwargs:
            del kwargs['legend_size']

        legend_title = kwargs.get('legend_title', None)
        if 'legend_title' in kwargs:
            del kwargs['legend_title']

        ax_result = func(*args, ax=ax, **kwargs)
        if rotate_xticklabels:
            ax_result.set_xticklabels(
                ax_result.get_xticklabels(),
                ha='right',
                rotation=rotate_xticklabels,
            )

        if not xticklabels:
            ax.set_xticks([])
            ax.set_xlabel(ax.get_xlabel(), labelpad=7.0)

        if not yticklabels:
            ax.set_yticks([])
            ax.set_ylabel(ax.get_ylabel(), labelpad=7.0)

        if legend_size is not None or legend_title is not None:
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, labels, prop={'size': legend_size}, title=legend_title)

        _try_save(file_path)
        return ax_result
    return wrapper


def set_suptitle(title):
    """Set suptitle for the plot.
    """
    plt.suptitle(title)


# Set plot preference which looks good to me.
def set_style(style='white', palette='deep', context='talk', font='FreeSans', font_scale=1.00, rcparams={'figure.figsize': (11.7, 8.27)}):
    """Set plot preference in a way that looks good to me.
    """
    import matplotlib.font_manager as font_manager

    styles = plt.style.available
    if 'dohlee' not in styles:
        if not os.path.exists(os.path.join(mpl.get_configdir(), 'stylelib')):
            os.makedirs(os.path.join(mpl.get_configdir(), 'stylelib'))

        request.urlretrieve(
            'https://sgp1.digitaloceanspaces.com/dohlee-bioinfo/dotfiles/dohlee.mplstyle',
            os.path.join(mpl.get_configdir(), 'stylelib', 'dohlee.mplstyle')
        )
        # Loading style files is done at import time, so we should explicitly reload style files.
        plt.style.reload_library()

    # Update font list.
    mpl_data_dir = os.path.dirname(mpl.matplotlib_fname())
    font_files = font_manager.findSystemFonts(os.path.join(mpl_data_dir, 'fonts', 'ttf'))
    font_list = font_manager.createFontList(font_files)
    font_manager.fontManager.ttflist.extend(font_list)

    sns.set(
        style=style,
        palette=palette,
        context=context,
        font=font,
        font_scale=font_scale,
        rc=rcparams
    )
    plt.style.use('dohlee')


def get_axis(preset=None, figsize=None, transpose=False, dpi=300):
    """Get plot axis with predefined/user-defined width and height.

    >>> get_axis(kind='extra-small')
    >>> get_axis(kind='small')
    >>> get_axis(kind='medium')
    >>> get_axis(kind='large')
    >>> get_axis(kind='extra-large')
    >>> get_axis(kind='wide')
    >>> get_axis(figsize=(7.2, 4.45))

    :param str preset: Use preset width and height. (extra-small, small, medium, large, extra-large)
    :param tuple figsize: Use user-defined width and height.
    :param bool transpose: Swap width and height.
    """
    w, h = 7.2, 4.45  # Nature double-column preset inches.
    assert (preset is None) or (figsize is None), 'You cannot use both preset and figsize argument.'
    assert not (preset is None and figsize is None), 'You should specify one of preset or figsize argument.'
    if preset is not None:
        if preset == 'extra-small':
            w, h = w / 4, h / 4
        elif preset == 'small':
            w, h = w / 3.5, h / 3.5
        elif preset == 'medium':
            w, h = w / 2, h / 2
        elif preset == 'large':
            w, h = w / 1.66, h / 1.66
        elif preset == 'extra-large':
            w, h = w / 1.33, h / 1.33
    else:
        w, h = figsize

    if transpose:
        w, h = h, w

    fig = plt.figure(figsize=(w, h), dpi=dpi)
    ax = fig.add_subplot(111)
    return ax


@_my_plot
def get_grid(shape=(1, 1), ax=None, hspace=None, wspace=None):
    """TODO
    """
    subplotspec = ax.get_subplotspec()
    return subplotspec.subgridspec(*shape, hspace=hspace, wspace=wspace)


def get_axis_from_grid(grid):
    """TODO
    """
    return plt.subplot(grid)


@_my_plot
def frequency(data, order=None, sort_by_values=False, dy=0.03, ax=None, **kwargs):
    """Plot frequency bar chart.

    >>> frequency([1, 2, 2, 3, 3, 3], order=[3, 1, 2], sort_by_values=True)

    :param list data: A list of elements.
    :param list order: A list of elements which represents the order of the elements to be plotted.
    :param bool sort_by_values: If True, the plot will be sorted in decreasing order of frequency values.
    :param float dy: Gap between a bar and its count label.
    :param pyplot-axis ax: Axis to draw the plot.
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

    ax = _get_ax_to_draw(ax)

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
        ax.text(
            x=x,
            y=count + dy,
            s=str(count),
            size='large',
            va='bottom',
            ha='center'
        )

    return ax


@_my_plot
def histogram(data, ax=None, **kwargs):
    """Draw a histogram.

    >>> histogram(data=data, ax=ax, lw=1.55)

    :param list data: A list containing values. Density of the values will be drawn as a histogram.
    :param axis ax: Matplotlib axis to draw the plot on.
    """
    ax.hist(data, color='black', ec='white', lw=1.33, **kwargs)
    return ax


@_my_plot
def boxplot(data, x, y, hue=None, ax=None, strip=False, **kwargs):
    """Draw a boxplot.

    >>> boxplot(data, x='species', y='sepal_length', strip=True)

    :param dataframe data: Dataframe for boxplot.
    :param str x: Column name representing x variable of the plot.
    :param str y: Column name representing y variable of the plot.
    :param axis ax: (Optional) Matplotlib axis to draw the plot on.
    :param bool strip: (default=False) Draw overlapped stripplot.
    """
    fliersize = 0 if strip else 5
    sns.boxplot(data=data, x=x, y=y, hue=hue, linewidth=1.33, flierprops={'marker': '.'}, fliersize=fliersize, ax=ax, **kwargs)
    if strip:
        sns.stripplot(data=data, x=x, y=y, hue=hue, jitter=.03, color='k', size=5, ax=ax)

    return ax


@_my_plot
def volcano(data, x, y, padj, label, cutoff=0.05, sample1=None, sample2=None, ax=None):
    """Draw a volcano plot.

    >>> volcano(data=data,
                x='log2FoldChange',
                y='pvalue',
                label='Gene_Symbol',
                cutoff=0.05,
                padj='padj',
                figsize=(10.8, 8.4))

    :param dataframe data: A dataframe resulting from DEG-discovery tool.
    :param str x: Column name denoting log2 fold change.
    :param str y: Column name denoting p-value. (Note that p-values will be log10-transformed, so they should not be transformed beforehand.)
    :param str padj: Column name denoting adjusted p-value.
    :param str label: Column name denoting gene identifier.
    :param float cutoff: (Optional) Adjusted p-value cutoff value to report significant DEGs.
    :param str sample1: (Optional) First sample name.
    :param str sample2: (Optional) Second sample name.
    :param axis ax: (Optional) Matplotlib axis to draw the plot on.
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

    if sample1 is not None and sample2 is not None:
        ax.set_xlabel(r'$log_2$FC ($log_{2}\frac{%s}{%s}$)' % (sample1.replace(' ', '\ '), sample2.replace(' ', '\ ')))
    else:
        ax.set_xlabel(r'$log_2$FC')
    ax.set_ylabel(r'$log_{10}$(p-value)')
    return ax


@_my_plot
def pca(data, labels=None, ax=None, **kwargs):
    '''Draw a simple principle component analysis plot of the data.

    :param matrix data: Input data. Numpy array recommended.
    :param list labels: (Optional) Corresponding labels to each datum. If specified, data points in the plot will be colored according to the label.
    :param axis ax: (Optional) Matplotlib axis to draw the plot on.
    :param kwargs: Any other keyword arguments will be passed onto matplotlib.pyplot.scatter.
    '''
    # Fit PCA and get pc's
    pca = PCA(n_components=2)
    pca.fit(data)
    pc = pca.transform(data)

    if labels is None:
        ax.scatter(x=pc[:, 0], y=pc[:, 1], **kwargs)

    else:
        # If labels are attached, color them in different colors
        labels = np.array(labels)
        for label in set(labels):
            toDraw = (labels == label)  # only draw these points this time

            ax.scatter(x=pc[toDraw, 0], y=pc[toDraw, 1], label=label, **kwargs)
            ax.legend(loc='best')

    # show explained variance ratio in the plot axes
    explainedVarianceRatio = pca.explained_variance_ratio_
    ax.set_xlabel('PC1 ({:.2%})'.format(explainedVarianceRatio[0]))
    ax.set_ylabel('PC2 ({:.2%})'.format(explainedVarianceRatio[1]))
    return ax


@_my_plot
def tsne(data, labels=None, ax=None, **kwargs):
    '''Draw a T-SNE analysis plot of the data.

    :param matrix data: Input data. Numpy array recommended.
    :param list labels: (Optional) Corresponding labels to each datum. If specified, data points in the plot will be colored according to the label.
    :param axis ax: (Optional) Matplotlib axis to draw the plot on.
    :param kwargs: Any other keyword arguments will be passed onto matplotlib.pyplot.scatter.
    '''
    # Fit T-SNE and get embeddings.
    tsne = TSNE(n_components=2)
    embeddings = tsne.fit(data)

    if labels is None:
        ax.scatter(x=embeddings[:, 0], y=embeddings[:, 1], **kwargs)

    else:
        # If labels are attached, color them in different colors
        labels = np.array(labels)
        for label in set(labels):
            toDraw = (labels == label)  # only draw these points this time

            ax.scatter(
                x=embeddings[toDraw, 0],
                y=embeddings[toDraw, 1],
                label=label,
                **kwargs
            )
            ax.legend(loc='best')
    return ax


@_my_plot
def mutation_signature(data, ax=None, **kwargs):
    def clear_spines(axis):
        # Hide spines.
        directions = ['top', 'right', 'bottom', 'left']
        for d in directions:
            axis.spines[d].set_visible(False)
        # Hide ticklabels.
        axis.set_xticklabels([])
        axis.set_yticklabels([])

    # (context, alt) -> index in 96-dim vector.
    def mut2ind(context, alt):
        ref = context[1]
        assert ref in ['C', 'T'], 'ref not in [C, T], ref: %s' % ref

        if ref == 'C':
            return ('AGT'.find(alt)) * 16 + c_contexts.index(context)
        else:
            return (3 + 'ACG'.find(alt)) * 16 + t_contexts.index(context)

    # Define trinucleotide contexts.
    c_contexts = [p + 'C' + n for (p, n) in itertools.product('ACGT', 'ACGT')]
    t_contexts = [p + 'T' + n for (p, n) in itertools.product('ACGT', 'ACGT')]

    # Convert data(Counter) as a 96-dimension vector.
    vectorized_data = np.zeros(96)
    for (context, alt), count in data.items():
        vectorized_data[mut2ind(context, alt)] = count
    vectorized_data = vectorized_data / vectorized_data.sum()

    #
    # Draw figure.
    #
    subplotspec = ax.get_subplotspec()
    grid = subplotspec.subgridspec(7, 6, hspace=0.03, wspace=0.03)

    # Draw annotation bars above the main plot.
    annotation_axes = [plt.subplot(grid[0, i]) for i in range(6)]
    annotations = ['C>A', 'C>G', 'C>T', 'T>A', 'T>C', 'T>G']
    colors = [
        (30, 190, 240),
        (5, 7, 9),
        (231, 39, 38),
        (202, 202, 202),
        (161, 207, 100),
        (238, 200, 197),
    ]

    for axis, annotation, color in zip(annotation_axes, annotations, colors):
        clear_spines(axis)
        axis.text(x=0.5, y=0.5, s=annotation, ha='center', fontsize='large')

        rect = patches.Rectangle((0, 0), width=1, height=0.33, facecolor=[x/255 for x in color], linewidth=0)
        axis.add_patch(rect)

    # Draw main plot.
    main_axis = plt.subplot(grid[1:, :])

    xticklabel_dx = 0.11
    main_axis.set_xticks(np.arange(0, 96, 1) + xticklabel_dx)
    main_axis.set_xticklabels(c_contexts * 3 + t_contexts * 3, rotation=90, fontsize='large', ha='left', va='top', fontdict={'family': 'Dejavu Sans Mono'})
    main_axis.tick_params(axis='x', pad=3)
    main_axis.set_xlim([0, 96])
    main_axis.spines['top'].set_visible(True)
    main_axis.spines['right'].set_visible(False)
    main_axis.yaxis.set_tick_params(size=5)
    main_axis.set_ylabel('Relative contribution')

    data = np.array([np.random.randint(0, 30) for _ in range(96)])
    data = data / data.sum()

    bars = main_axis.bar(np.arange(0, 96, 1) + 0.5, vectorized_data, width=0.5)
    for i in range(6):
        start, end = 16 * i, 16 * (i+1)
        r, g, b = colors[i]
        for bar in bars[start:end]:
            bar.set_facecolor([r / 255, g / 255, b / 255])


@_my_plot
def linear_regression(x, y, regression=True, ax=None, color='k'):
    """TODO
    """
    # Perform linear regression.
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

    ax.scatter(x, y, s=5, color=color)
    x_extent = np.array(ax.get_xlim())
    ax.plot(x_extent, slope * x_extent + intercept, lw=1, color=color)

    legend = ax.legend(
        labels=['$R^2$ = %.3f, p = %.3g' % (r_value ** 2, p_value)],
        loc='best',
        fontsize='small',
        handlelength=0,
        handletextpad=0
    )
    for item in legend.legendHandles:
        item.set_visible(False)


@_my_plot
def m_bias(mbias_data, ax=None):
    mbias_data['CpG Methylation (%)'] = mbias_data.apply(axis=1, func=lambda row: row.nMethylated / (row.nMethylated + row.nUnmethylated) * 100)

    fig = plt.figure(figsize=(8.4 * 2.2, 8.4), dpi=300)

    commands = []
    for ax_ind, strand in enumerate(['OT', 'OB'], 1):
        bounds = []

        ax = fig.add_subplot(1, 2, ax_ind)
        ax.set_ylim([0, max(70, max(mbias_data['CpG Methylation (%)']))])
        ax.set_title(strand, fontsize='x-large')
        ax.set_xlabel('Position along mapped read')
        ax.set_ylabel('CpG Methylation (%)')

        tmp = mbias_data[(mbias_data.Strand == strand)]
        for read in tmp.Read.unique():
            this_data = tmp[tmp.Read == read]

            x = this_data['Position'].values
            y = this_data['CpG Methylation (%)'].values
            ax.plot(x, y, lw=1, label='Read %d' % read)
            ax.set_xlim([0, max(x)])

            for i in range(0, max(x), 10):
                ax.axvline(x=i, linestyle='--', linewidth=0.66, color='grey', alpha=0.33)

            lower_bound, upper_bound, cis = analysis._get_inclusion_bounds(this_data)
            ax.axvline(x=lower_bound, linestyle='--', linewidth=1, color='red', alpha=0.66)
            ax.axvline(x=upper_bound, linestyle='--', linewidth=1, color='red', alpha=0.66)

            ax.legend(fontsize='large', loc='best')

            bounds.append(lower_bound)
            bounds.append(upper_bound)

        if len(bounds) == 2:
            for _ in range(2):
                bounds.append(0)
        commands.append('--%s %s' % (strand, ','.join(map(str, bounds))))


@_my_plot
def stacked_bar_chart(data, x, y, ax=None, sort=False, reverse=True, sort_by=None, group=None, group_order=None, group_label=True):
    """TODO
    """
    ax = _get_ax_to_draw(ax)
    ax.margins(x=5e-3)
    if not isinstance(y, list):
        y = list(y)

    x_values = data[x].values
    y_values = data[y].values.T
    bottoms = data[y].values.cumsum(axis=1).T
    x_positions = np.arange(len(x_values))

    if group:
        groups = data[group].values
    if sort:
        if sort_by is None:
            # If sort_by parameter is not specified, sort by the sum of the heights of stacked bars.
            order_mask = np.argsort(data[y].values.sum(axis=1))
        else:
            assert sort_by in y, 'Column name in sort_by parameter should appear in y parameter.'
            order_mask = np.argsort(data[sort_by].values)
        if reverse:
            order_mask = order_mask[::-1]
        x_values = x_values[order_mask]
        y_values = y_values[:, order_mask]
        bottoms = bottoms[:, order_mask]
        if group:
            groups = groups[order_mask]

    if group is not None:
        group_mask = []
        group_count = Counter()

        group_order = group_order if group_order is not None else data[group].unique()
        for g in group_order:
            for i, g_tmp in enumerate(groups):
                if g_tmp == g:
                    group_count[g] += 1
                    group_mask.append(i)

        # Sort x_values, y_values and bottoms to reflect group assignments.
        # Convert to list to use `insert` method.
        x_values = list(x_values[group_mask])
        y_values = y_values[:, group_mask]
        bottoms = bottoms[:, group_mask]

        # Insert empty space between each of the two groups.
        insert_index = 0
        group_label_position = []
        for i, g in enumerate(group_order):
            group_label_position.append(insert_index + group_count[g] / 2)
            insert_index += group_count[g]
            x_values.insert(insert_index, '')
            num_rows = y_values.shape[0]
            y_values = np.hstack([y_values[:, :insert_index], np.zeros((num_rows, 1)), y_values[:, insert_index:]])
            bottoms = np.hstack([bottoms[:, :insert_index], np.zeros((num_rows, 1)), bottoms[:, insert_index:]])
            insert_index += 1

        x_positions = np.arange(len(x_values))

    ax.bar(x_positions, y_values[0], label=y[0])
    for i in range(1, len(y_values)):
        ax.bar(x_positions, y_values[i], bottom=bottoms[i-1], label=y[i])

    ax.legend()
    if group is not None and group_label:
        ax.set_xticks(group_label_position)
        ax.set_xticklabels(group_order, ha='center')
    else:
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_values)

    ax.set_ylim([0, ax.get_ylim()[1]])
    ax.tick_params('y', length=5, width=1, which='major')
    return ax

@_my_plot
def umap(data, labels=None, ax=None, **kwargs):
    '''Draw a UMAP embedding plot of the data.

    :param matrix data: Input data. Numpy array recommended.
    :param list labels: (Optional) Corresponding labels to each datum. If specified, data points in the plot will be colored according to the label.
    :param axis ax: (Optional) Matplotlib axis to draw the plot on.
    :param kwargs: Any other keyword arguments will be passed onto matplotlib.pyplot.scatter.
    '''
    # Apply UMAP and get embeddings.
    reducer = UMAP()
    embeddings = reducer.fit_transform(data)

    if labels is None:
        ax.scatter(x=embeddings[:, 0], y=embeddings[:, 1], **kwargs)

    else:
        # If labels are attached, color them in different colors
        labels = np.array(labels)
        for label in set(labels):
            toDraw = (labels == label)  # only draw these points this time

            ax.scatter(
                x=embeddings[toDraw, 0],
                y=embeddings[toDraw, 1],
                label=label,
                **kwargs
            )
            ax.legend(loc='best')
    return ax


@_my_plot
def dimensionality_reduction(data, labels, ax=None):
    ax = get_axis(figsize=(20, 4))
    grid = get_grid((1, 5), ax=ax)
    axes = [get_axis_from_grid(grid[i]) for i in range(5)]

    common_args = dict(
        data=data,
        labels=labels,
        # FIXME: HARD-CODED legend size.
        legend_size='medium',
        xticklabels=False,
        yticklabels=False,
    )

    pca(ax=axes[0], title='PCA', **common_args)
    tsne(ax=axes[1], title='T-SNE', **common_args)
    umap(ax=axes[2], title='UMAP', **common_args)


@_my_plot
def line(y, data=None, x=None, hue=None, ax=None, agg='mean', **kwargs):
    if data is None:
        if x is None:
            ax.plot(y, **kwargs)
        else:
            ax.plot(x, y, **kwargs)
    else:  # `data` is not None.
        if hue is None:
            if x is None:
                ax.plot(data[y], **kwargs)
            if x is not None:
                ax.plot(data[x], data[y], **kwargs)
        else:
            # `data` is not None, and `hue` is not None.
            # We should aggregate data by `hue` column and plot each line separately.
            if x is None:
                aggregated = data.groupby([hue, x, y]).agg(agg)
