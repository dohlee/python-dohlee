import sys

import numpy as np
import pandas as pd

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
