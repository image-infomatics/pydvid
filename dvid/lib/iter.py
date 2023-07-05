from collections.abc import Iterator, Iterable

import numpy as np
import h5py
import pandas as pd

def iter_batches(it, batch_size):
    """
    Iterator.
    
    Consume the given iterator/iterable in batches and
    yield each batch as a list of items.
    
    The last batch might be smaller than the others,
    if there aren't enough items to fill it.
    
    If the given iterator supports the __len__ method,
    the returned batch iterator will, too.
    """
    if hasattr(it, '__len__'):
        return _iter_batches_with_len(it, batch_size)
    else:
        return _iter_batches(it, batch_size)


class _iter_batches:
    def __init__(self, it, batch_size):
        self.base_iterator = it
        self.batch_size = batch_size
                

    def __iter__(self):
        return self._iter_batches(self.base_iterator, self.batch_size)
    

    def _iter_batches(self, it, batch_size):
        if isinstance(it, (pd.DataFrame, pd.Series)):
            for batch_start in range(0, len(it), batch_size):
                yield it.iloc[batch_start:batch_start+batch_size]
            return
        elif isinstance(it, (list, np.ndarray, h5py.Dataset)):
            for batch_start in range(0, len(it), batch_size):
                yield it[batch_start:batch_start+batch_size]
            return
        else:
            if not isinstance(it, Iterator):
                assert isinstance(it, Iterable)
                it = iter(it)
    
            while True:
                batch = []
                try:
                    for _ in range(batch_size):
                        batch.append(next(it))
                except StopIteration:
                    return
                finally:
                    if batch:
                        yield batch


class _iter_batches_with_len(_iter_batches):
    def __len__(self):
        return int(np.ceil(len(self.base_iterator) / self.batch_size))