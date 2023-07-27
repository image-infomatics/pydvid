from functools import partial
from multiprocessing.pool import ThreadPool
from multiprocessing import get_context

from .progress import tqdm_proxy

def compute_parallel(func, iterable, chunksize=1, threads=0, processes=0, ordered=None,
                     leave_progress=False, total=None, initial=0, starmap=False, show_progress=None,
                     context=None, shutdown_delay=0.15, **pool_kwargs):
    """
    Use the given function to process the given iterable in a ThreadPool or process Pool,
    showing progress using tqdm.

    Args:
        func:
            The function to process each item with.

        iterable:
            The items to process.

        chunksize:
            Send items to the pool in chunks of this size.

        threads:
            If given, use a ThreadPool with this many threads.

        processes
            If given, use a multiprocessing Pool with this many processes.
            Note: When using a process pool, your function and iterable items must be pickleable.

        ordered:
            Must be either True, False, or None:

            - If True, process the items in order, and return results
              in the same order as provided in the input.

            - If False, process the items as quickly as possible,
              meaning that some results will be presented out-of-order,
              depending on how long they took to complete relative to the
             other items in the pool.

            - If None, then the items will be processed out-of-order,
              but the results will be reordered to correspond to the original
              input order before returning.

        total:
            Optional. Specify the total number of tasks, for progress reporting.
            Not necessary if your iterable defines __len__.

        initial:
            Optional. Specify a starting value for the progress bar.

        starmap:
            If True, each item should be a tuple, which will be unpacked into
            the arguments to the given function, like ``itertools.starmap()``.

        show_progress:
            If True, show a progress bar.
            By default, only show a progress bar if ``iterable`` has more than one element.

        context:
            In Python, process pools can be created via 'fork', 'spawn', or 'forkserver'.
            Spawn is more robust, but comes with certain requirements
            (such as your main code being shielded within a __main__ conditional).
            See the Python multiprocessing docs for details.

        pool_kwargs:
            keyword arguments to pass to the underlying Pool object,
            such as ``initializer`` or ``maxtasksperchild``.
    """
    assert not bool(threads) or not bool(processes), \
        "Specify either threads or processes, not both"
    assert context in (None, 'fork', 'spawn', 'forkserver')
    assert ordered in (True, False, None)
    reorder = (ordered is None)

    # Pick a pool implementation
    if threads:
        pool = ThreadPool(threads, **pool_kwargs)
    elif processes:
        pool = get_context(context).Pool(processes, **pool_kwargs)
    else:
        pool = _DummyPool()

    if total is None and hasattr(iterable, '__len__'):
        total = len(iterable)

    # Pick a map() implementation
    if not threads and not processes:
        f_map = map
    elif ordered:
        f_map = partial(pool.imap, chunksize=chunksize)
    else:
        f_map = partial(pool.imap_unordered, chunksize=chunksize)

    # If we'll need to reorder the results,
    # then pass an index into (and out of) the function,
    # which we'll use to sort the results afterwards.
    if reorder:
        iterable = enumerate(iterable)

    # By default we call the function directly,
    # but the 'reorder' or 'starmap' options require wrapper functions.
    if reorder and starmap:
        func = partial(_idx_passthrough_apply_star, func)
    elif reorder and not starmap:
        func = partial(_idx_passthrough, func)
    elif not reorder and starmap:
        func = partial(_apply_star, func)

    if show_progress is None:
        if hasattr(iterable, '__len__') and len(iterable) == 1:
            show_progress = False
        else:
            show_progress = True

    with pool:
        iter_results = f_map(func, iterable)
        results_progress = tqdm_proxy(iter_results, initial=initial, total=total, leave=leave_progress, disable=not show_progress)
        try:
            with results_progress:
                # Here's where the work is actually done, i.e. during iteration.
                results = []
                for item in results_progress:
                    results.append(item)
        except KeyboardInterrupt as ex:
            # If the user killed the job early, provide the results
            # that have completed so far via an exception attribute.
            if reorder:
                results.sort(key=itemgetter(0))
                results = [r for (_, r) in results]

            # IPython users can access the exception via sys.last_value
            raise KeyboardInterruptWithResults(results, total or '?') from ex
        finally:
            # I see hangs here from time to time during normal operation,
            # even when no exception is raised (I think).
            # I suspect this is either related to my use of 'fork' as a multiprocecssing mode,
            # (which is generally frowned upon), or perhaps it's a bug in multiprocessing itself.
            # In any case, I'll try to combat the issue via:
            #
            #   1. Fight possible race conditions with a slight delay after the
            #      last item completes.
            #
            #   2. I also tried calling pool.close() here, but it seemed to cause
            #      deadlocks when terminate() is called.
            #
            # For reference, here is an example traceback for a hanged pool:
            #
            #     Thread 2854017 (idle): "Dask-Default-Threads-2848426-2"
            #         poll (multiprocessing/popen_fork.py:28)
            #         wait (multiprocessing/popen_fork.py:48)
            #         join (multiprocessing/process.py:140)
            #         _terminate_pool (multiprocessing/pool.py:617)
            #         __call__ (multiprocessing/util.py:224)
            #         terminate (multiprocessing/pool.py:548)
            #         __exit__ (multiprocessing/pool.py:623)
            #         compute_parallel (neuclease/util/util.py:723)
            #         ...
            #
            # Relevant discussions that might be related to this deadlock issue:
            # - https://sefiks.com/2021/07/05/handling-hang-in-python-multiprocessing/
            # - https://bugs.python.org/issue33997 and PR https://github.com/python/cpython/pull/8009
            # - https://stackoverflow.com/questions/65620077
            #
            #
            # Note: I could call pool.close() here, but I think that creates a
            #        deadlock in terminate() so I don't do that anymore.
            if shutdown_delay:
                time.sleep(shutdown_delay)
            pool.terminate()

    if reorder:
        results.sort(key=itemgetter(0))
        results = [r for (_, r) in results]

    return results


class KeyboardInterruptWithResults(KeyboardInterrupt):
    def __init__(self, partial_results, total_items):
        super().__init__()
        self.partial_results = partial_results
        self.total_items = total_items

    def __str__(self):
        return f'{len(self.partial_results)}/{self.total_items} results completed (see sys.last_value)'

class _DummyPool:
    def __enter__(self):
        pass

    def __exit__(self, *args):
        pass

    def close(self):
        pass

    def terminate(self):
        pass


def _apply_star(func, arg):
    return func(*arg)


def _idx_passthrough(func, idx_arg):
    idx, arg = idx_arg
    return idx, func(arg)


def _idx_passthrough_apply_star(func, idx_arg):
    idx, arg = idx_arg
    return idx, func(*arg)

