import pybedtools

def extend_bed(bed, bp, direction='both'):
    """Extend each interval in bed file with specified basepairs.
    If bp is a list, each interval will be extended bp[0] toward upstream, and
    bp[1] toward downstream.

    Note: Be careful! This method does not deal with strandedness of the interval.

    :param (pybedtools.BedTool) bed: Bed object to be manipulated.
    :param (int, list) bp: Each interval will be extended according to bp.
    :param (str) direction: Direction of extension. Note that if bp is a list, direction will be fixed to 'both' ('up', 'down', 'both')
    """
    if isinstance(bp, list):
        assert len(bp) in [1, 2], 'Check parameter `bp`.'
        if len(bp) == 2:
            direction = 'both'
        else:
            bp = [bp[0], bp[0]]
    else:
        bp = [bp, bp]

    upstream_extension = 0 if direction == 'down' else bp[0]
    downstream_extension = 0 if direction == 'up' else bp[1]

    bed_strings = []
    for interval in bed:
        start = max(interval.start - upstream_extension, 0)
        end = interval.end + downstream_extension

        if start < 0:
            continue

        bed_strings.append(
            '%s %d %d %s %s %s' % (interval.chrom, start, end, interval.name, interval.score, interval.strand)
        )

    return pybedtools.BedTool('\n'.join(bed_strings), from_string=True)
