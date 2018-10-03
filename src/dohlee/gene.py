import mygene

from collections import OrderedDict

HOMO_SAPIENS_SPECIES_ID = 9606


def get_first_item(items):
    """Return the first item of items. If 'items' is a single value, just return it.

    :param list/value items: A list of items or a single value.

    :returns: The first item of items.
    """
    return mygene.alwayslist(items)[0]


def _return_best_results(raw_results):
    best_result = OrderedDict()
    for k, v in reversed(raw_results):
        best_result[k] = v

    result = list(reversed(best_result.values()))
    return result[0] if len(result) == 1 else result

def ensg2symbol(ensembl_ids):
    """Convert Ensembl gene ids into gene symbols.

    :param list ensembl_ids: A list of Ensembl IDs to be converted.

    :returns: A list of HGNC symbols, which is the result of best conversion of given Ensembl IDs.
    """
    mg = mygene.MyGeneInfo()
    query_results = mg.getgenes(ensembl_ids, fields='symbol', species=HOMO_SAPIENS_SPECIES_ID)
    raw_results = [(query_result['query'], query_result['symbol']) for query_result in query_results if 'symbol' in query_result]

    return _return_best_results(raw_results)


def symbol2ensg(symbols=None):
    """Convert gene symbols into Ensembl gene ids.

    :param list symbols: A list of HGNC symbols to be converted.

    :returns: A list of Ensembl gene symbols(ENSG symbols).
    """
    mg = mygene.MyGeneInfo()
    query_results = mg.querymany(symbols, scopes='symbol', fields='ensembl.gene', species=HOMO_SAPIENS_SPECIES_ID)
    raw_results = [(query_result['query'], get_first_item(query_result['ensembl'])['gene'])
                  for query_result in query_results
                  if 'ensembl' in query_result]

    return _return_best_results(raw_results)
