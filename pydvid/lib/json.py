
import math
from collections.abc import Mapping, Sequence

import ujson
from numba import jit

def convert_nans(o, nullval="NaN", _c=None):
    """
    Traverse the given collection-of-collections and
    replace all NaN values with the string "NaN".
    Also converts numpy arrays into lists.
    Intended for preprocessing objects before JSON serialization.
    """
    _c = _c or {}

    if isinstance(o, float) and math.isnan(o):
        return nullval
    elif isinstance(o, np.number):
        if np.isnan(o):
            return nullval
        return o.tolist()
    elif isinstance(o, (str, bytes)) or not isinstance(o, (Sequence, Mapping)):
        return o

    # Even though this function is meant mostly for JSON,
    # so we aren't likely to run into self-referencing
    # or cyclical object graphs, we handle that case by keeping
    # track of the objects we've already processed.
    if id(o) in _c:
        return _c[id(o)]

    if isinstance(o, np.ndarray):
        ret = []
        _c[id(o)] = ret
        ret.extend([convert_nans(x, nullval, _c) for x in o.tolist()])
    elif isinstance(o, Sequence):
        ret = []
        _c[id(o)] = ret
        ret.extend([convert_nans(x, nullval, _c) for x in o])
    elif isinstance(o, Mapping):
        ret = {}
        _c[id(o)] = ret
        ret.update({k: convert_nans(v, nullval, _c) for k,v in o.items()})
    else:
        raise RuntimeError(f"Can't handle {type(o)} object: {o}")

    return ret

# used in json_dump(), above
_convert_nans = convert_nans


def unsplit_json_int_lists(json_text):
    """
    When pretty-printing json data, it will split all lists across several lines.
    For small lists of integers (such as [x,y,z] points), that may not be desirable.
    This function "unsplits" all lists of integers and puts them back on a single line.

    Example:
        >>> s = '''\\
        ... {
        ...   "body": 123,
        ...   "supervoxel": 456,
        ...   "coord": [
        ...     123,
        ...     456,
        ...     789
        ...   ],
        ... }
        ... '''

        >>> u = unsplit_json_int_lists(s)
        >>> print(u)
        {
        "body": 123,
        "supervoxel": 456,
        "coord": [123,456, 781],
        }

    """
    json_text = re.sub(r'\[\s+(\d+),', r'[\1,', json_text)
    json_text = re.sub(r'\n\s*(\d+),', r' \1,', json_text)
    json_text = re.sub(r'\n\s*(\d+)\s*\]', r' \1]', json_text)
    return json_text


def write_json_list(objects, f):
    """
    Like json.dump(), but writes each item to its own line (no indentation).
    """
    assert isinstance(objects, list)

    def _impl(f):
        f.write('[\n')
        for s in objects[:-1]:
            ujson.dump(s, f)
            f.write(',\n')
        ujson.dump(objects[-1], f)
        f.write('\n]')

    if isinstance(f, str):
        with open(f, 'w') as fp:
            _impl(fp)
    else:
        _impl(f)


def gen_json_objects(f, batch_size=None, parse=True):
    """
    Generator.
    
    Given a file containing a JSON list-of-objects,
    parse the objects one-by-one and iterate over them.
    
    Args:
        f:
            A file containing a JSON document which must be a list-of-objects.
            Must be an actual on-disk file (or a path to one),
            becuase it will be memory-mapped and therefore must have a fileno(). 

        batch_size:
            If provided, the objects will be yielded in groups
            (lists) of the specified size.

        parse:
            If True, each json object will be parsed and yielded as a dict.
            Otherwise, the raw text of the object is returned.
    """
    m = np.memmap(f, mode='r')
    it = map(bytes, _gen_json_objects(m))
    
    if parse:
        it = map(ujson.loads, it)
        
    if batch_size is None:
        yield from it
    else:
        yield from iter_batches(it, batch_size)


@jit(nopython=True, nogil=True)
def _gen_json_objects(text_array):
    """
    Generator.
    
    Parse a JSON list-of-objects one at a time,
    without reading in the entire file at once.
    
    Each object is yielded and then discarded.
    
    Warnings:
        - The input MUST be valid JSON, and specifically must be a list-of-objects.
          Any other input results in undefined behavior and/or errors.
        - Strings containing curly braces are not supported.
          (The document must not contain any curly braces except for the ones
          defining actual JSON objects.)
        
    Args:
        text_array:
            A np.array (dtype == np.uint8) which, when interpreted as text,
            contains a list-of-dicts JSON document.
    
    Yields:
        Every object in the document, one at a time.
    """
    nest_level = 0
    cur_start = 0
    cur_stop = 0
    for i, c in enumerate(text_array):
        if c == b'{'[0]:
            if nest_level == 0:
                cur_start = i
            nest_level += 1
        if c == b'}'[0]:
            nest_level -= 1
            if nest_level == 0:
                cur_stop = i+1
                yield text_array[cur_start:cur_stop]