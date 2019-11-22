import matplotlib.pyplot as plt

from . import util

def lineplot(data, y, x='pos', name=None, line_kwargs={}, show_coord=False, ax=None):
    """
    """
    if ax is None:
        ax = util.get_default_ax()
    
    sorted_d = data.sort_values(x)
    ax.plot(sorted_d[x].values, sorted_d[y].values, **line_kwargs)
    
    ymin, ymax = util.get_y_extent(sorted_d[y].values)
    ax = util.format_axis(ax, name, show_coord, ymin, ymax)

    return ax

def barplot(data, y, x='pos', name=None, bar_kwargs={}, show_coord=False, ax=None):
    """
    """
    if ax is None:
        ax = util.get_default_ax()

    sorted_d = data.sort_values(x)
    ax.bar(sorted_d[x].values, sorted_d[y].values, width=1.0, lw=0, **bar_kwargs)

    ymin, ymax = util.get_y_extent(sorted_d[y].values)
    ax = util.format_axis(ax, name, show_coord, ymin, ymax)
    
    return ax

def bam(bam_fp, chrom, start, end, name=None, kind='bar', plot_kwargs={}, show_coord=False, ax=None):
    """
    """
    d = util.parse_bam(bam_fp, chrom, start, end)
    if kind == 'bar':
        ax = barplot(d, y='coverage', x='pos', name=name, bar_kwargs=plot_kwargs, show_coord=show_coord, ax=ax)
    elif kind == 'line':
        ax = lineplot(d, y='coverage', x='pos', name=name, line_kwargs=plot_kwargs, show_coord=show_coord, ax=ax)
    
    return ax

def wiggle(wig_fp, chrom, start, end, name=None, kind='line', plot_kwargs={}, show_coord=False, ax=None):
    """
    """
    d = util.parse_wiggle(wig_fp, chrom, start, end)
    if kind == 'line':
        ax = lineplot(d, y='value', x='pos', name=name, line_kwargs=plot_kwargs, show_coord=show_coord, ax=ax)
    elif kind == 'bar':
        ax = barplot(d, y='value', x='pos', name=name, bar_kwargs=plot_kwargs, show_coord=show_coord, ax=ax)
    
    return ax
