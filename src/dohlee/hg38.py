from pyensembl import EnsemblRelease

ENSEMBL_RELEASE_VERSION = 87


def gene_by_id(*args, **kwargs):
    genome = EnsemblRelease(ENSEMBL_RELEASE_VERSION)
    return genome.gene_by_id(*args, **kwargs)


def genes_by_name(*args, **kwargs):
    genome = EnsemblRelease(ENSEMBL_RELEASE_VERSION)
    return genome.genes_by_name(*args, **kwargs)


def genes_at_locus(*args, **kwargs):
    genome = EnsemblRelease(ENSEMBL_RELEASE_VERSION)
    return genome.genes_at_locus(*args, **kwargs)


def gene_names_at_locus(*args, **kwargs):
    genome = EnsemblRelease(ENSEMBL_RELEASE_VERSION)
    return genome.gene_names_at_locus(*args, **kwargs)


def symbol2ensg(*args):
    genome = EnsemblRelease(ENSEMBL_RELEASE_VERSION)
    return genome.gene_ids_of_gene_name(*args)[0]
