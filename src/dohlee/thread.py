import multiprocessing as mp
from tqdm import tqdm, tqdm_notebook


def imap_helper(args):
    """
    Helper function for imap.
    This is needed since built-in multiprocessing library does not have `istarmap` function.
    If packed arguments are passed, it unpacks the arguments and pass through the function.
    Otherwise, it just pass the argument through the given function.

    Attributes:
        args: Tuple of two arguments, user-defined function and arguments to pass through.
    """
    assert len(args) == 2
    func = args[0]
    # Arguments are packed as a list or a tuple, so pass *args.
    if isinstance(args[1], list) or isinstance(args[1], tuple):
        return func(*(args[1]))
    # Otherwise, just pass the argument.
    else:
        return func(args[1])


def threaded(func, params, processes, progress=False, progress_type='tqdm'):
    """
    Generate results of the function with given parameters with threads.

    Attributes:
        func (function): Function to be executed.
        params (iterable): A list of parameters.
        processes (int): Number of processes to work on.
        progress (bool): if True, show progress bar.
        progress_type (str): 'tqdm' or 'tqdm_notebook' can be used.
    """
    n_params = len(list(params))
    with mp.Pool(processes=processes) as p:
        if progress:
            if progress_type not in ['tqdm', 'tqdm_notebook']:
                # If given progresstype is not supported,
                # fall back to tqdm.tqdm.
                progress_type = 'tqdm'

            if progress_type == 'tqdm':
                # Use tqdm.tqdm.
                for result in tqdm(p.imap(imap_helper, [(func, p) for p in params]), total=n_params):
                    yield result
            elif progress_type == 'tqdm_notebook':
                # Use tqdm.tqdm_notebook.
                for result in tqdm_notebook(p.imap(imap_helper, [(func, p) for p in params]), total=n_params):
                    yield result
        else:
            for result in p.imap(imap_helper, [(func, p) for p in params]):
                yield result
