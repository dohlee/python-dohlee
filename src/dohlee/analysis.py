import os
import sys
import pybedtools
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import defaultdict, Counter
from dohlee import bed as bed_util
from dohlee import util
from dohlee import plot

from dohlee.thread import threaded

def cv(series):
    if series.mean() == 0:
        warnings.warn('Mean of series is 0 when computing coefficient of variance.')

    return series.std() / series.mean()

def _confidence_interval(meth, unmeth):
    """Compute Agresti-Coull confidence interval.
    Reference:
    https://github.com/dpryan79/MethylDackel/blob/084d926c90c5275d5520fa3346feed144d36bf48/svg.c
    """
    n = meth + unmeth
    zz = 10.8275661707 # qnorm(0.9995)^2
    z = 3.2905267315 # qnorm(0.9995)
    n_dot = n + zz
    p_dot = (1.0 / n_dot) * (meth + 0.5 * zz)

    s = z * np.sqrt((p_dot / n_dot) * (1 - p_dot))
    return p_dot - s , p_dot + s

def _get_inclusion_bounds(df):
    num_position = df.Position.max()
    low, high, middle = int(num_position * 0.2), int(num_position * 0.8), num_position // 2

    # Step 1: Get the average methylation of the middle 60% of the array.
    average_methylation = df[(df.Position > low) & (df.Position <= high)]['CpG Methylation (%)'].mean() / 100

    # Step 2: Determine the min/max of the confidence intervals of the middle 60%.
    cis = [_confidence_interval(row.nMethylated, row.nUnmethylated) for i, row in df[(df.Position > low) & (df.Position <= high)].iterrows()]
    min_ci, max_ci = min([l for (l, h) in cis]), max([h for (l, h) in cis])

    # Get lower inclusion bound.
    lower_bound = 0
    for i in range(middle, 0, -1):
        this_df = df[df.Position == i]
        meth, unmeth, value = this_df.nMethylated.values[0], this_df.nUnmethylated.values[0], this_df['CpG Methylation (%)'].values[0] / 100
        ci = _confidence_interval(meth, unmeth)

        if (
            (ci[1] < average_methylation and value < min_ci and abs(value - average_methylation) > 0.05)
            or (ci[0] > average_methylation and value > max_ci and abs(value - average_methylation) > 0.05)
        ):
            lower_bound = i + 1
            break

    # Get upper inclusion bound.
    upper_bound = 0
    for i in range(middle + 1, num_position + 1):
        this_df = df[df.Position == i]
        meth, unmeth, value = this_df.nMethylated.values[0], this_df.nUnmethylated.values[0], this_df['CpG Methylation (%)'].values[0] / 100
        ci = _confidence_interval(meth, unmeth)

        if (
            (ci[1] < average_methylation and value < min_ci and abs(value - average_methylation) > 0.05)
            or (ci[0] > average_methylation and value > max_ci and abs(value - average_methylation) > 0.05)
        ):
            upper_bound = i - 1
            break

    whole_cis = [_confidence_interval(row.nMethylated, row.nUnmethylated) for i, row in df.iterrows()]
    return lower_bound, upper_bound, whole_cis

def methyldackel_extract_bounds():
    """TODO
    """
    mbias_data = pd.read_table(sys.stdin)
    mbias_data['CpG Methylation (%)'] = mbias_data.apply(axis=1, func=lambda row: row.nMethylated / (row.nMethylated + row.nUnmethylated) * 100)

    commands = []
    for ax_ind, strand in enumerate(['OT', 'OB'], 1):
        bounds = []
        tmp = mbias_data[(mbias_data.Strand == strand)]
        for read in tmp.Read.unique():
            this_data = tmp[tmp.Read == read]

            lower_bound, upper_bound, cis = _get_inclusion_bounds(this_data)
            bounds.append(lower_bound)
            bounds.append(upper_bound)

        if len(bounds) == 2:
            for _ in range(2):
                bounds.append(0)
        commands.append('--%s %s' % (strand, ','.join(map(str, bounds))))

    print(' '.join(commands))


def default_get_bin(pos, start, end, bp_extension, outer_bin_size=50, inner_bin_count=100):
    if start <= pos < start + bp_extension[0]:
        x = pos - start
        bin_number = x // outer_bin_size
    elif start + bp_extension[0] <= pos < end - bp_extension[1]:
        binsize = (end - start - sum(bp_extension)) / inner_bin_count
        x = pos - (start + bp_extension[0])
        bin_number = int(x / binsize) + bp_extension[0] // outer_bin_size
    else:
        x = end - pos
        bin_number = x // outer_bin_size + bp_extension[0] // outer_bin_size + inner_bin_count
    return bin_number


def aggregate_methylation_landscapes(
    annotation_bed_fp,
    methylation_bed_fps,
    condition_fp,
    temp_file_dir,
    SAMPLE_NAME_COL,
    CONDITION_COL,
    bp_extension=None,
    POS_COL='Position',
    METH_LEVEL_COL='Methylation level',
    BIN_COL='Bin',
    agg='mean',
    binning_func=None,
    outer_bin_size=50,
    inner_bin_count=100,
    extension_direction='both',
    threads=4,
    threaded_kws={'progress': False},
):
    """Plot averaged methylation values across the regions which are annotated in the bed file.
    
    :param str annotation_bed_fp: Path to a BED file defining the region of interest.
    :param list methylation_bed_fps: A list of paths of BED files defining the meth-level of each CpG position. 
    :param str condition_fp: Path to a two-column file which represents conditions of samples.
    :param str temp_file_dir: Path to a directory to save caches.
    :param str SAMPLE_NAME_COL: Name of the column denoting IDs of samples.
    :param str CONDITION_COL: Name of the column denoting conditions of samples.
    :param int bp_extension: Amount of extension.
    :param str POS_COL: (default='Position') Name of the column denoting CpG positions.
    :param str METH_LEVEL_COL: (default='Methylation level') Name of the column denoting methylation levels.
    :param str BIN_COL: (default='Bin') Name of the column denoting bins.
    :param str agg: (default='mean') Aggregation method. e.g. 'mean', 'var', 'min', 'max', 'cv'
    :param func binning_func: If possible, pass custom function that represents your binning strategy.
    :param int outer_bin_size: Size of each bin in outer extended regions.
    :param int inner_bin_count: Number of bins in inner regions, which are originally defined in the annotation BED file.
    :param str extension_direction: Extend view toward 'both'/'up'/'down' direction.
    :param int threads: Number of threads to use.
    :param dict threaded_kws: Keyword arguments passed to dohlee.threaded function.
    """
    binning_func = default_get_bin  if binning_func is None else binning_func
    agg          = cv               if agg == 'cv'          else agg

    condition = pd.read_table(condition_fp)
    # Sanity check of the condition file.
    assert SAMPLE_NAME_COL in condition.columns.values, \
        'Column representing sample names %s is not in condition file.' % SAMPLE_NAME_COL

    # Sort and extend annotations with specified amount of bp's.
    ann_bed          = pybedtools.BedTool(annotation_bed_fp).sort()
    interval_lengths = [(iv.end - iv.start) for iv in ann_bed]

    # If extension parameter is not given, set it as the most common interval length in BED file.
    most_common_interval_length = util.most_common_value(Counter(interval_lengths))
    if bp_extension is None:
        bp_extension = [most_common_interval_length] * 2
    
    # Extend each of the intervals in given BED file with the amount given.
    extended_ann_bed = bed_util.extend_bed(ann_bed, bp=bp_extension, direction=extension_direction)

    # Prepare parameters for multithreaded calls.
    params = [(bed, extended_ann_bed, temp_file_dir, binning_func, bp_extension, outer_bin_size, inner_bin_count, SAMPLE_NAME_COL, POS_COL, METH_LEVEL_COL, BIN_COL) for bed in methylation_bed_fps]
    # Run multithreaded runs of aggregating (by computing average) methylation levels.
    aggregated_dfs = list(threaded(func=aggregate_intersecting_methylation_levels, params=params, processes=threads, **threaded_kws))
    final_df = pd.concat(aggregated_dfs).merge(condition, on=SAMPLE_NAME_COL, how='inner')

    # Draw plot.
    ax = plot.get_axis(preset='wide')
    plt.style.use('dohlee')

    condition_mean_val_dict = defaultdict(list)
    for (condition, bin_num), data in final_df.groupby([CONDITION_COL, BIN_COL]).agg(agg).iterrows():
        condition_mean_val_dict[condition].append(data[METH_LEVEL_COL])

    # TODO: Allow user to specify order of conditions to be plotted.
    for condition, y in condition_mean_val_dict.items():
        ax.plot(y, lw=0.5, label=condition)

    ax.set_ylabel('Methylation level')
    ax.tick_params(left=True, length=3, width=0.5, axis='y')

    # Thinner spines.
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)

    # Show legends.
    ax.legend(prop={'size': 'large'})
    return ax

def aggregate_intersecting_methylation_levels(
    methylation_bed_fp,
    extended_ann_bed,
    temp_file_dir,
    binning_func,
    bp_extension,
    outer_bin_size,
    inner_bin_count,
    SAMPLE_NAME_COL,
    POS_COL,
    METH_LEVEL_COL,
    BIN_COL,
):
    sample_name = util.strip_ext(methylation_bed_fp)
    temp_file = os.path.join(temp_file_dir, '%s_aggregated.tsv' % sample_name)

    if not os.path.exists(temp_file):
        methylation_bed = pybedtools.BedTool(methylation_bed_fp).sort()
        intersection_bed = methylation_bed.intersect(extended_ann_bed, loj=True) \
                                          .filter(lambda iv: iv.fields[6] != '.') \
                                          .saveas(os.path.join(temp_file_dir, '%s_tmp.bed' % sample_name))

        data = {
            SAMPLE_NAME_COL: [],
            POS_COL: [],
            METH_LEVEL_COL: [],
            BIN_COL: [],
        }

        for interval in intersection_bed:
            # Start and end position of the overlapping annotation interval.
            try:
                start, end = int(interval.fields[7]), int(interval.fields[8])
            except:
                with open(os.path.join(temp_file_dir, 'error.txt'), 'w') as outFile:
                    print(interval, file=outFile)
                raise IndexError 

            data[SAMPLE_NAME_COL].append(sample_name)
            data[POS_COL].append(interval.start)
            data[METH_LEVEL_COL].append(interval.score)
            data[BIN_COL].append(binning_func(interval.start, start, end, bp_extension, outer_bin_size, inner_bin_count))

        data = pd.DataFrame(data)
        data[METH_LEVEL_COL] = data[METH_LEVEL_COL].astype(np.float32)
        data.to_csv(temp_file, sep='\t', index=False, header=True)
    else:
        data = pd.read_table(temp_file)
        assert SAMPLE_NAME_COL in data.columns.values

    aggregated = data.groupby([BIN_COL]).agg({
        SAMPLE_NAME_COL: 'first',
        POS_COL: 'first',
        METH_LEVEL_COL: 'mean',
    })
    aggregated[BIN_COL] = aggregated.index
    return aggregated
