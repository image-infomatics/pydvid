
import networkx as nx


def uuids_match(uuid1: str, uuid2: str):
    """
    Return True if the two uuids are the equivalent.
    
    >>> assert uuids_match('abcd', 'abcdef') == True
    >>> assert uuids_match('abc9', 'abcdef') == False
    """
    assert uuid1 and uuid2, "Empty UUID"
    n = min(len(uuid1), len(uuid2))
    return (uuid1[:n] == uuid2[:n])

def find_root(g: nx.DiGraph, start=None):
    """
    Find the root node in a tree, given as a nx.DiGraph,
    tracing up the tree starting with the given start node.
    """
    if start is None:
        start = next(iter(g.nodes()))
    parents = [start]
    while parents:
        root = parents[0]
        parents = list(g.predecessors(parents[0]))
    return root


def tree_to_dict(tree: nx.DiGraph, root, display_fn=str, *, _d=None):
    """
    Convert the given tree (nx.DiGraph) into a dict,
    suitable for display via the asciitree module.

    Args:
        tree:
            nx.DiGraph
        root:
            Where to start in the tree (ancestors of this node will be ignored)
        display_fn:
            Callback used to convert node values into strings, which are used as the dict keys.
        _d:
            Internal use only.
    """
    if _d is None:
        _d = {}
    d_desc = _d[display_fn(root)] = {}
    for n in tree.successors(root):
        tree_to_dict(tree, n, display_fn, _d=d_desc)
    return _d

