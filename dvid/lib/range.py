#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import numpy as np


class ndrange:
    """
    Generator.

    Like np.ndindex, but accepts start/stop/step instead of
    assuming that start is always (0,0,0) and step is (1,1,1).
    
    Example:
    
    >>> for index in ndrange((1,2,3), (10,20,30), step=(5,10,15)):
    ...     print(index)
    (1, 2, 3)
    (1, 2, 18)
    (1, 12, 3)
    (1, 12, 18)
    (6, 2, 3)
    (6, 2, 18)
    (6, 12, 3)
    (6, 12, 18)
    
    See also: ``ndindex_array()``
    """

    def __init__(self, start, stop=None, step=None):
        if stop is None:
            stop = start
            start = (0,)*len(stop)
    
        if step is None:
            step = (1,)*len(stop)
    
        assert len(start) == len(stop) == len(step), \
            f"tuple lengths don't match: ndrange({start}, {stop}, {step})"

        self.start = start
        self.stop = stop
        self.step = step
    
    def __iter__(self):
        return product(*starmap(range, zip(self.start, self.stop, self.step)))

    def __len__(self):
        span = (np.array(self.stop) - self.start)
        step = np.array(self.step)
        return np.prod( (span + step-1) // step )


def ndrange_array(start, stop=None, step=None):
    """
    Like np.ndindex, but accepts start/stop/step instead of
    assuming that start is always (0,0,0) and step is (1,1,1),
    and returns an array instead of an iterator.
    """
    if stop is None:
        stop = start
        start = (0,)*len(stop)

    start, stop = box = np.array((start, stop))
    aligned_box = box - start
    if step is None:
        # Step is implicitly 1
        shape = aligned_box[1]
        return start + ndindex_array(*shape)
    else:
        shape = round_coord(aligned_box[1], step, 'up') // step
        return start + step * ndindex_array(*shape)


def ndindex_array(*shape, dtype=np.int32):
    """
    Like np.ndindex, but returns an array.
    
    numpy has no convenience function for this, and won't any time soon.
    https://github.com/numpy/numpy/issues/1234#issuecomment-545990743
    
    Example:
    
        >>> ndindex_array(3,4)
        array([[0, 0],
               [0, 1],
               [0, 2],
               [0, 3],
               [1, 0],
               [1, 1],
               [1, 2],
               [1, 3],
               [2, 0],
               [2, 1],
               [2, 2],
               [2, 3]])
    """
    return np.indices(shape, dtype=dtype).reshape(len(shape), -1).transpose()


class NumpyConvertingEncoder(json.JSONEncoder):
    """
    Encoder that converts numpy arrays and scalars
    into their pure-python counterparts.
    
    (No attempt is made to preserve bit-width information.)
    
    Usage:
    
        >>> d = {"a": np.arange(3, dtype=np.uint32)}
        >>> json.dumps(d, cls=NumpyConvertingEncoder)
        '{"a": [0, 1, 2]}'
    """
    def default(self, o):
        if isinstance(o, (np.ndarray, np.number, np.bool_)):
            return o.tolist()
        return super().default(o)
