import pysam

def get_coverages(bam, chrom, start, end, strict=False):
    """Returns numpy array of coverages in given region.
    If strict=True, trim the coverages lying outside the given region.
    Note that the region should be formatted in zero-based coordinate.
    """
    pileup = pysam.AlignmentFile(bam).pileup(chrom, start, end)
    coverage_dict = {col.pos:col.n for col in pileup if start <= col.pos < end}
    
    if strict:
        positions = list(range(start, end))
        return positions, [coverage_dict.get(pos, 0) for pos in positions]
    else:
        positions = list(coverage_dict.keys())
        min_pos = min(min(positions), start)
        max_pos = max(max(positions), end)
        positions = list(range(min_pos, max_pos + 1))
        return positions, [coverage_dict.get(pos, 0) for pos in positions]
