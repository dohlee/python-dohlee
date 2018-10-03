import dohlee.thread as thread

from nose.tools import raises

def process_single_parameter_task(x):
    return x**2


def process_multiple_parameter_task(x, y, z):
    return x * y * z


def imap_helper_single_parameter_test():
    assert thread.imap_helper((process_single_parameter_task, 2)) == 4


def imap_helper_multiple_parameter_test():
    assert thread.imap_helper((process_multiple_parameter_task, [1, 2, 3])) == 6


def threaded_single_parameter_with_no_progress_test():
    tasks = [1, 2, 3]
    results = list(thread.threaded(func=process_single_parameter_task, params=tasks, processes=2, progress=False))
    assert results == [1, 4, 9]


def threaded_multiple_parameter_with_no_progress_test():
    tasks = [(1, 2, 3), (2, 3, 4), (3, 4, 5)]
    results = list(thread.threaded(func=process_multiple_parameter_task, params=tasks, processes=2, progress=False))
    assert results == [6, 24, 60]


def threaded_single_parameter_with_other_progress_test():
    tasks = [1, 2, 3]
    results = list(thread.threaded(func=process_single_parameter_task, params=tasks, processes=2, progress=True, progress_type='other'))
    assert results == [1, 4, 9]


def threaded_single_parameter_with_tqdm_progress_test():
    tasks = [1, 2, 3]
    results = list(thread.threaded(func=process_single_parameter_task, params=tasks, processes=2, progress=True))
    assert results == [1, 4, 9]


@raises(ImportError)
def threaded_single_parameter_with_no_progress_test():
    tasks = [1, 2, 3]
    results = list(thread.threaded(func=process_single_parameter_task, params=tasks, processes=2, progress=True, progress_type='tqdm_notebook'))
