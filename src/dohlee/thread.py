import multiprocessing as mp
from tqdm import tqdm, tqdm_notebook


def imap_helper(args):
    """
    TODO
    """
    assert len(args) == 2
    func = args[0]
    return func(*(args[1]))


def threaded(func, params, processes, progress=False, progress_type='tqdm'):
    """
    Generate results of the function with given parameters with threads.

    Attributes:
        func (function): Function to be executed.
        params (iterable): A list of parameters.
        processes (int): Number of processes to work on.
        progress (bool): if True, show progress bar.
        progress_type (str): 'tqdm' - Default tqdm.tqdm will be used for progress bar.
            'tqdm_notebook' - tqdm.tqdm_notebook will be used for progress bar.
    """
    def star_func(args):
        return func(*args)

    with mp.Pool(processes=processes) as p:
        if progress:
            if progress_type not in ['tqdm', 'tqdm_notebook']:
                # If given progresstype is not supported,
                # fall back to tqdm.tqdm.
                progress_type = 'tqdm'

            if progress_type == 'tqdm':
                # Use tqdm.tqdm.
                for result in tqdm(p.imap(imap_helper, [(func, p) for p in params]), total=len(list(params))):
                    yield result
            elif progress_type == 'tqdm_notebook':
                # Use tqdm.tqdm_notebook.
                for result in tqdm_notebook(p.imap(imap_helper, [(func, p) for p in params]), total=len(list(params))):
                    yield result
        else:
            for result in p.imap(imap_helper, [(func, p) for p in params]):
                yield result
