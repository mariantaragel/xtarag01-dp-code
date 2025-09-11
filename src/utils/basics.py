"""
Originally wrote sometime around 2018.
Panos Achlioptas (https://optas.github.io)
"""

import multiprocessing as mp
from multiprocessing import Pool


def parallel_apply(iterable, func, n_processes=None):
    """ Apply func in parallel to chunks of the iterable based on multiple processes.
    :param iterable:
    :param func: simple function that does not change the state of global variables.
    :param n_processes: (int) how many processes to split the data over
    :return:
    """
    n_items = len(iterable)
    if n_processes is None:
        n_processes = min(4 * mp.cpu_count(), n_items)
    pool = Pool(n_processes)
    chunks = int(n_items / n_processes)
    res = []
    for data in pool.imap(func, iterable, chunksize=chunks):
        res.append(data)
    pool.close()
    pool.join()
    return res
