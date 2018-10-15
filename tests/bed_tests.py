import pybedtools
import dohlee.bed as bed_util

from nose.tools import raises

def extend_bed_up_test():
    # chr1  100 200
    bed = pybedtools.BedTool('tests/data/test.bed')
    result_bed = bed_util.extend_bed(
        bed=bed,
        bp=10,
        direction='up',
    )

    assert result_bed[0].chrom == 'chr1'
    assert result_bed[0].start == 90
    assert result_bed[0].end == 200


def extend_bed_down_test():
    # chr1  100 200
    bed = pybedtools.BedTool('tests/data/test.bed')
    result_bed = bed_util.extend_bed(
        bed=bed,
        bp=10,
        direction='down',
    )

    assert result_bed[0].chrom == 'chr1'
    assert result_bed[0].start == 100
    assert result_bed[0].end == 210


def extend_bed_both_test():
    # chr1  100 200
    bed = pybedtools.BedTool('tests/data/test.bed')
    result_bed = bed_util.extend_bed(
        bed=bed,
        bp=10,
        direction='both',
    )

    assert result_bed[0].chrom == 'chr1'
    assert result_bed[0].start == 90
    assert result_bed[0].end == 210


def extend_bed_clipping_test():
    # chr1  100 200
    bed = pybedtools.BedTool('tests/data/test.bed')
    result_bed = bed_util.extend_bed(
        bed=bed,
        bp=150,
        direction='both',
    )

    assert result_bed[0].chrom == 'chr1'
    assert result_bed[0].start == 0
    assert result_bed[0].end == 350


@raises(AssertionError)
def extend_bed_invalid_bp_failing_test():
    # chr1  100 200
    bed = pybedtools.BedTool('tests/data/test.bed')
    result_bed = bed_util.extend_bed(
        bed=bed,
        bp=[1, 2, 3],
    )


def extend_bed_single_element_list_bp_test():
    # chr1  100 200
    bed = pybedtools.BedTool('tests/data/test.bed')
    result_bed = bed_util.extend_bed(
        bed=bed,
        bp=[10],
    )

    assert result_bed[0].chrom == 'chr1'
    assert result_bed[0].start == 90
    assert result_bed[0].end == 210


def extend_bed_double_element_list_bp_test():
    # chr1  100 200
    bed = pybedtools.BedTool('tests/data/test.bed')
    result_bed = bed_util.extend_bed(
        bed=bed,
        bp=[10, 20],
    )

    assert result_bed[0].chrom == 'chr1'
    assert result_bed[0].start == 90
    assert result_bed[0].end == 220
