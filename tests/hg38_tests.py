from nose.tools import assert_equal
from sklearn import datasets

import dohlee.hg38 as hg38

def gene_by_id_test():
    gene = hg38.gene_by_id('ENSG00000121879')
    assert_equal(gene.gene_name, 'PIK3CA')


def genes_by_name_test():
    genes = hg38.genes_by_name('TP53')
    assert_equal(genes[0].gene_id, 'ENSG00000141510')


def genes_at_locus_test():
    genes = hg38.genes_at_locus(contig='chr17', position=7670000)
    assert_equal(genes[0].gene_name, 'TP53')


def gene_names_at_locus_test():
    gene_names = hg38.gene_names_at_locus(contig='chr17', position=7670000)
    assert_equal(gene_names[0], 'TP53')


def symbol2ensg_test():
    symbol = 'TP53'
    ensg = hg38.symbol2ensg('TP53')
    assert_equal(ensg, 'ENSG00000141510')
