from collections import namedtuple
import pysam
import numpy as np

class Region(namedtuple('Region', ['chrom', 'start', 'end'])):
    def __new__(cls, chrom, start, end):
        if start >= end:
            raise ValueError('Invalid genomic region %s:%d-%d' % (chrom, start, end))
        
        return super(Region, cls).__new__(cls, chrom, start, end)
    
    def merge(self, r):
        """Merge with another region.
        """
        return merge_regions(self, r)
    
    def has_overlap_with(self, r):
        """Returns True if the region has overlap with r,
        otherwise False.
        """
        return has_overlap(self, r)
    
    def __len__(self):
        return self.end - self.start + 1

    def __str__(self):
        return '%s:%d-%d' % (self.chrom, self.start, self.end)

def merge_regions(r1, r2):
    """Merge two regions.
    """
    assert r1.chrom == r2.chrom, \
        'Unable to merge regions. Contigs are incompatible: %s and %s' % (r1.chrom, r2.chrom)

    start_new, end_new = min(r1.start, r2.start), max(r1.end, r2.end)
    return Region(r1.chrom, start_new, end_new)

def has_overlap(r1, r2):
    if r1.chrom != r2.chrom:
        return False

    a0, a1 = r1.start, r1.end
    b0, b1 = r2.start, r2.end
    
    return a0 <= b1 and b0 <= a1

def fetch_annotation(annotation, chrom, start, end):
    """Returns a generator of annoation rows for given genomic region.
    """
    return pysam.TabixFile(annotation).fetch(chrom, start, end, parser=pysam.asGTF)

def get_gene_ids_at_region(annotation, chrom=None, start=None, end=None, region=None):
    assert (chrom is None) or (region is None), "Please don't specify chrom and region at the same time."
    if region is not None:
        chrom, start, end = region.chrom, region.start, region.end
    fetched = fetch_annotation(annotation, chrom, start, end)
    return [row.gene_id for row in fetched if row.feature == 'gene']

def get_transcript_ids_at_region(annotation, chrom=None, start=None, end=None):
    assert (chrom is None) or (region is None), "Please don't specify chrom and region at the same time."
    if region is not None:
        chrom, start, end = region.chrom, region.start, region.end
    fetched = fetch_annotation(annotation, chrom, start, end)
    return [row.transcript_id for row in fetched if row.feature == 'transcript']

def annotate(table, db):
    if 'chrom' not in table.columns:
        raise ValueError("Column 'chrom' should be specified in the table.")
    if 'start' not in table.columns:
        raise ValueError("Column 'start' should be specified in the table.")
    if 'end' not in table.columns:
        raise ValueError("Column 'end' should be specified in the table.")

    for db_name, db_path in db.items():
        tbx = pysam.TabixFile(db_path)
        values = []
        for record in table.itertuples():
            hits = [read.name for read in tbx.fetch(record.chrom, record.start, record.end, parser=pysam.asBed())]
            # hits = []
            # for read in tbx.fetch(record.chrom, record.start, record.end, parser=pysam.asBed()):
                # if read.start <= record.start and record.end <= read.end:
                    # hits.append(read.name)

            v = ';'.join(hits) if hits else np.nan
            values.append(v)

        table[db_name] = values
    return table

if __name__ == '__main__':
    r1 = Region('chr1', 1000, 2000)
    r2 = Region('chr1', 1500, 3000)
    r3 = Region('chr2', 1000, 2000)
    
    print('len r1 =', len(r1))
    print('len r2 =', len(r2))
    print('len r3 =', len(r3))

    print('Merge(r1, r2) = ', r1.merge(r2))
    print('Merge(r1, r2) = ', merge_regions(r1, r2))

    print('has_overlap(r1, r2) = ', r1.has_overlap_with(r2))
    print('has_overlap(r1, r2) = ', has_overlap(r1, r2))
    print('has_overlap(r1, r3) = ', r1.has_overlap_with(r3))
    print('has_overlap(r1, r3) = ', has_overlap(r1, r3))
    print('has_overlap(r2, r3) = ', r2.has_overlap_with(r3))
    print('has_overlap(r2, r3) = ', has_overlap(r2, r3))
