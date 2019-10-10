from collections import namedtuple

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