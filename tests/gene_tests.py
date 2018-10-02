from nose.tools import assert_equal

import dohlee.gene as gene


def given_single_gene_ensg2symbol_test():
    ensg = 'ENSG00000141510'
    symbol = gene.ensg2symbol(ensg)
    assert_equal(symbol, 'TP53')


def given_multiple_gene_ensg2symbol_test():
    ensgs = ['ENSG00000121879', 'ENSG00000141510']
    symbols = gene.ensg2symbol(ensgs)
    assert_equal(symbols, ['PIK3CA', 'TP53'])


def given_single_gene_symbol2ensg_test():
    symbol = 'TP53'
    ensg = gene.symbol2ensg(symbol)
    assert_equal(ensg, 'ENSG00000141510')


def given_multiple_gene_symbol2ensg_test():
    symbols = ['PIK3CA', 'TP53']
    ensgs = gene.symbol2ensg(symbols)
    assert_equal(ensgs, ['ENSG00000121879', 'ENSG00000141510'])
