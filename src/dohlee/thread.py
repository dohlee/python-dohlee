import multiprocessing as mp
from tqdm import tqdm


def imap_helper(args):
    """
    TODO
    """
    assert len(args) == 2
    func = args[0]
    return func(*(args[1]))


def threaded(func, params, processes, progress=False):
    """
    TODO
    """
    def star_func(args):
        return func(*args)

    with mp.Pool(processes=processes) as p:
        if progress:
            for result in tqdm(p.imap(imap_helper, [(func, p) for p in params]), total=len(params)):
                yield result
        else:
            for result in p.imap(imap_helper, [(func, p) for p in params]):
                yield result
