import re
import tqdm
import pysam
import itertools
import pandas as pd
import numpy as np

from dohlee.genome import Region
from collections import defaultdict, Counter


def get_binary_string_representation(read):
    """Returns a binary representation string of methylation state of a read.
    Unmethylated state = 0, methylated state = 1.

    Requires Bismark-aligned read.
    """
    return ''.join([['0', '1'][c == 'Z'] for c in read.get_tag('XM') if c in ['z', 'Z']])

def character_indices(string, characters):
    i = []
    for character in characters:
        i.extend([m.start() for m in re.finditer(character, string)])
    
    return list(sorted(i))

def get_cpg_coordinates(read, paired=False):
    start = read.reference_start
    methylation_string = read.get_tag('XM')
    
    if not paired:
        if not read.is_reverse:
            return tuple(start + i + 1 for i in character_indices(methylation_string, ['z', 'Z']))
        else:
            return tuple(start + i for i in character_indices(methylation_string, ['z', 'Z']))
        
    else:
        if read.flag in [99, 147]:
            return tuple(start + i + 1 for i in character_indices(methylation_string, ['z', 'Z']))
        else:
            return tuple(start + i for i in character_indices(methylation_string, ['z', 'Z']))
    
def is_reverse(read, paired=False):
    return read.is_reverse if not paired else (read.flag in [99, 147])

def is_concordant(read_bin):
    return len(set(read_bin)) == 1

def pdr(reads_binarized):
    num_discordant = sum(not is_concordant(read_bin) for read_bin in reads_binarized)
    return num_discordant / len(reads_binarized)
    
def methylation_level(reads_binarized):
    num_meth, num_total = 0, 0
    for read_bin in reads_binarized:
        num_meth += sum(CpG == '1' for CpG in read_bin)
        num_total += len(read_bin)
    
    return num_meth / num_total

def methylation_entropy(reads_binarized):
    counts = Counter(reads_binarized)
    b = len(reads_binarized[0])
    N = sum(counts.values())
    return 1/b * sum(-n_i / N * np.log2(n_i / N) for n_i in counts.values())

def epipolymorphism(reads_binarized):
    counts = Counter(reads_binarized)
    N = sum(counts.values())
    ps = [n_i / N for n_i in counts.values()]
    return 1 - sum(p**2 for p in ps)

def get_concordancy_counters1(coordinates, reads_binarized):
    concordant_counter, discordant_counter = Counter(), Counter()
    for read_binarized in reads_binarized:
        indexed_cpgs = [(idx, cpg) for idx, cpg in zip(coordinates, read_binarized)]
        for (idx1, cpg1), (idx2, cpg2) in itertools.combinations(indexed_cpgs, 2):
            dist = abs(idx2 - idx1)
            if cpg1 == cpg2:
                concordant_counter[dist] += 1
            else:
                discordant_counter[dist] += 1

    return concordant_counter, discordant_counter

def get_concordancy_counters2(coordinates, reads_binarized):
    concordant_counter, discordant_counter = Counter(), Counter()
    for read_binarized in reads_binarized:
        indexed_cpgs = [(idx, cpg) for idx, cpg in zip(coordinates, read_binarized)]
        for (idx1, cpg1), (idx2, cpg2) in zip(indexed_cpgs[:-1], indexed_cpgs[1:]):
            dist = abs(idx2 - idx1)
            if cpg1 == cpg2:
                concordant_counter[dist] += 1
            else:
                discordant_counter[dist] += 1

    return concordant_counter, discordant_counter

def is_chr_prepended(path):
    for read in pysam.AlignmentFile(path):
        return read.reference_name.startswith('chr')


def summarize_rrbs(path, file, contigs=None, paired=False, min_CpGs=None, min_depth=None):
    current_region = None
    
    cpgs_reads_dict = defaultdict(list)
    summarized = defaultdict(list)

    bam = pysam.AlignmentFile(path)

    # Setup default contig names if they are not given.
    if contigs is None:
        contigs = list(map(str, range(1, 23))) + ['X', 'Y']
        if is_chr_prepended(path):
            contigs = ['chr' + contig for contig in contigs]

    for contig in contigs:
        count = 0
        
        for read in tqdm.tqdm(bam.fetch(contig), desc=contig):
            read_region = Region(read.reference_name, read.reference_start, read.reference_end)

            if current_region is None:
                current_region = read_region
                cpgs_reads_dict[get_cpg_coordinates(read, paired)].append(read)
                continue

            if current_region.has_overlap_with(read_region):
                current_region = current_region.merge(read_region)
                cpgs_reads_dict[get_cpg_coordinates(read, paired)].append(read)
                
            else:
                # Take the largest bunch of reads.
                best_CpGs = max(cpgs_reads_dict.keys(), key=lambda x: len(cpgs_reads_dict[x]))
                best_reads = cpgs_reads_dict[best_CpGs]
                best_reads_binarized = [get_binary_string_representation(read) for read in best_reads]
                
                # Check if the number of CpGs are all the same.
                assert len(set([len(read_bin) for read_bin in best_reads_binarized])) == 1
                
                meet_cpg = (not min_CpGs) or (min_CpGs and len(best_reads_binarized[0]) >= min_CpGs)
                meet_depth = (not min_depth) or (min_depth and len(best_reads_binarized) >= min_depth)

                # We only deal with RRBS regions with more than 1 CpGs.
                if len(best_reads_binarized[0]) > 0 and meet_cpg and meet_depth:
                    # Summarize.
                    summarized['chrom'].append(current_region.chrom)
                    summarized['CpGs'].append(';'.join([current_region.chrom + ':' + str(CpG) for CpG in best_CpGs]))
                    summarized['num_forward'].append(sum(not is_reverse(read, paired=paired) for read in best_reads))
                    summarized['num_reverse'].append(sum(is_reverse(read, paired=paired) for read in best_reads))
                    summarized['num_concordant'].append(sum(is_concordant(read_bin) for read_bin in best_reads_binarized))
                    summarized['num_discordant'].append(sum(not is_concordant(read_bin) for read_bin in best_reads_binarized))
                    summarized['depth'].append(len(best_reads))
                    summarized['pdr'].append(pdr(best_reads_binarized))
                    summarized['num_CpGs'].append(len(best_reads_binarized[0]))
                    summarized['methylation_level'].append(methylation_level(best_reads_binarized))
                    summarized['methylation_entropy'].append(methylation_entropy(best_reads_binarized))
                    summarized['epipolymorphism'].append(epipolymorphism(best_reads_binarized))
                
                current_region = read_region
                cpgs_reads_dict = defaultdict(list)
                cpgs_reads_dict[get_cpg_coordinates(read, paired)].append(read)
        
    summarized = pd.DataFrame(summarized)
    summarized.to_csv(file, index=False)
    return summarized

def get_concordancy_table(path, file, contigs=None, paired=False, min_depth=10, min_num_cpgs=3):
    current_region = None
    
    cpgs_reads_dict = defaultdict(list)
    summarized = defaultdict(list)

    bam = pysam.AlignmentFile(path)

    # Setup default contig names if they are not given.
    if contigs is None:
        contigs = list(map(str, range(1, 23))) + ['X', 'Y']
        if is_chr_prepended(path):
            contigs = ['chr' + contig for contig in contigs]

    concordant_counter, discordant_counter = Counter(), Counter()
    for contig in contigs:
        count = 0
        
        for read in tqdm.tqdm(bam.fetch(contig), desc=contig):
            read_region = Region(read.reference_name, read.reference_start, read.reference_end)

            if current_region is None:
                current_region = read_region
                cpgs_reads_dict[get_cpg_coordinates(read, paired)].append(read)
                continue

            if current_region.has_overlap_with(read_region):
                current_region = current_region.merge(read_region)
                cpgs_reads_dict[get_cpg_coordinates(read, paired)].append(read)
                
            else:
                # Take the largest bunch of reads.
                best_CpGs = max(cpgs_reads_dict.keys(), key=lambda x: len(cpgs_reads_dict[x]))
                best_reads = cpgs_reads_dict[best_CpGs]
                best_reads_binarized = [get_binary_string_representation(read) for read in best_reads]
                
                # Check if the number of CpGs are all the same.
                assert len(set([len(read_bin) for read_bin in best_reads_binarized])) == 1
                
                # We only deal with RRBS regions with more than or equal to min_depth reads,
                # and more than or equal to min_num_cpgs CpGs.
                depth, num_cpgs = len(best_reads_binarized), len(best_reads_binarized[0])
                if depth >= min_depth and num_cpgs >= min_num_cpgs:
                    # Get concordancy counters.
                    c, d = get_concordancy_counters2(best_CpGs, best_reads_binarized)
                    concordant_counter += c
                    discordant_counter += d
                
                current_region = read_region
                cpgs_reads_dict = defaultdict(list)
                cpgs_reads_dict[get_cpg_coordinates(read, paired)].append(read)

    result = defaultdict(list) 
    for distance in sorted(concordant_counter.keys()):
        c, d = concordant_counter[distance], discordant_counter[distance]
        t = c + d

        result['distance'].append(distance)
        result['mean_concordance'].append(c / t)
        result['num_concordant'].append(c)
        result['num_total'].append(t)
    
    result = pd.DataFrame(result)
    result.to_csv(file, index=False)
    return result
