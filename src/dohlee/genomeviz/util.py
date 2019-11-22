import pysam

import pyBigWig as pbw
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

MAX_DEPTH = 1000000
Y_EXTENTS = [-1000, -5000, -250, -100, -50,
            -25, -10, -5, -2.5, -1, -0.5, 0, 0.5,
            1, 2.5, 5, 10, 25, 50, 100, 250, 500, 1000]

def parse_bam(bam_fp, chrom, start, end):
    """
    """
    data = defaultdict(list)

    bam = pysam.AlignmentFile(bam_fp)
    for pileupcolumn in bam.pileup(chrom, start, end, truncate=True, max_depth=MAX_DEPTH):
        data['chrom'].append(chrom)
        data['pos'].append(pileupcolumn.pos)
        data['coverage'].append(pileupcolumn.n)

    # Assert the dataframe has start and end positions.
    positions = set(data['pos'])
    for pos in [start, end]:
        if pos not in positions:
            data['chrom'].append(chrom) 
            data['pos'].append(pos)
            data['coverage'].append(0)
    
    return pd.DataFrame(data).sort_values('pos').reset_index()

def parse_wiggle(wig_fp, chrom, start, end):
    """
    """
    data = defaultdict(list)
    bw = pbw.open(wig_fp)
    
    values = bw.values(chrom, start, end)
    for pos, v in zip(range(start, end), values):
        if not np.isnan(v) and v != 0:
            data['chrom'].append(chrom)
            data['pos'].append(pos)
            data['value'].append(v)

    positions = set(data['pos'])
    for pos in [start, end]:
        if pos not in positions:
            data['chrom'].append(chrom)
            data['pos'].append(pos)
            data['value'].append(0)

    return pd.DataFrame(data).sort_values('pos').reset_index()

def get_y_extent(v):
    """Automatically determines appropriate y extents of a track.
    """
    v_min, v_max = np.min(v), np.max(v)
    y_min = max([y for y in Y_EXTENTS if y <= v_min])
    y_max = min([y for y in Y_EXTENTS if y >= v_max])

    return y_min, y_max

def format_axis(ax, name, show_coord, ymin, ymax, color=None):
    
    if not show_coord:
        ax.set_xticks([])

    ax.set_ylabel(name, rotation='horizontal', fontdict={'ha': 'right', 'va': 'center', 'fontsize': 10, 'color': ['k', color][color is not None]})
    ax.set_ylim(ymin, ymax)
    ax.set_yticks([ymin, ymax])

    ax.get_yticklabels()[0].set_verticalalignment('bottom')
    ax.get_yticklabels()[1].set_verticalalignment('top')
    ax.get_yticklabels()[0].set_horizontalalignment('right')
    ax.get_yticklabels()[1].set_horizontalalignment('right')

    ax.tick_params(axis='y', direction='in', width=2, labelsize=9)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    return ax
   
def get_default_ax():
    fig = plt.figure(figsize=(10, 1))
    ax = fig.add_subplot(111)

    return ax
