import logging
import numpy as np
from io import BytesIO

import vigra

from .lib.box import box_to_slicing, round_box
from .lib.grid import boxes_from_grid
from .lib.progress import tqdm_proxy
from .node import fetch_instance_info
from . import dvid_api_wrapper

logger = logging.getLogger(__name__)

@dvid_api_wrapper
def fetch_raw(server, uuid, instance, box_zyx, throttle=False, scale=0, *, dtype=np.uint8, session=None):
    """
    Fetch raw array data from an instance that contains voxels.
    
    Args:
        server:
            dvid server, e.g. 'emdata3:8900'
        
        uuid:
            dvid uuid, e.g. 'abc9'
        
        instance:
            dvid instance name, e.g. 'grayscale'

        box_zyx:
            The bounds of the volume to fetch in the coordinate system for the requested scale.
            Given as a pair of coordinates (start, stop), e.g. [(0,0,0), (10,20,30)], in Z,Y,X order.
            The box need not be block-aligned.
        
        throttle:
            If True, passed via the query string to DVID, in which case DVID might return a '503' error
            if the server is too busy to service the request.
            It is your responsibility to catch DVIDExceptions in that case.

        scale:
            For instance types that support it, specify the scale to fetch.
            Note:
                Some voxels data instances do not support a 'scale' parameter.
                Instead, by convention, we create multiple data instances with a suffix indicating the scale.
                For instance, 'grayscale', 'grayscale_1', 'grayscale_2', etc.
    
        dtype:
            The datatype of the underlying data instance.
            Must match the data instance dtype, e.g. np.uint8 for instances of type uint8blk.
    
    Returns:
        np.ndarray
    """
    box_zyx = np.asarray(box_zyx)
    assert np.issubdtype(box_zyx.dtype, np.integer), \
        f"Box has the wrong dtype.  Use an integer type, not {box_zyx.dtype}"
    assert box_zyx.shape == (2,3)

    params = {}
    if throttle:
        params['throttle'] = 'true'

    if scale:
        params['scale'] = int(scale)

    shape_zyx = (box_zyx[1] - box_zyx[0])
    shape_str = '_'.join(map(str, shape_zyx[::-1]))
    offset_str = '_'.join(map(str, box_zyx[0, ::-1]))

    r = session.get(f'{server}/api/node/{uuid}/{instance}/raw/0_1_2/{shape_str}/{offset_str}', params=params)
    r.raise_for_status()

    if len(r.content) != np.prod(shape_zyx) * np.dtype(dtype).itemsize:
        info = fetch_instance_info(server, uuid, instance)
        typename = info["Base"]["TypeName"]
        msg = ("Buffer from DVID is the wrong length for the requested array.\n"
               "Did you pass the correct dtype for this instance?\n"
               f"Instance '{instance}' has type '{typename}', and you passed dtype={np.dtype(dtype).name}")
        raise RuntimeError(msg)

    a = np.frombuffer(r.content, dtype=dtype)
    return a.reshape(shape_zyx)


@dvid_api_wrapper
def post_raw(server, uuid, instance, offset_zyx, volume, throttle=False, mutate=True, *, session=None):
    offset_zyx = np.asarray(offset_zyx)
    assert offset_zyx.shape == (3,)
    assert np.issubdtype(offset_zyx.dtype, np.integer), \
        f"Offset has the wrong dtype.  Use an integer type, not {offset_zyx.dtype}"

    params = {}
    if throttle:
        params['throttle'] = 'true'        
    if mutate:
        params['mutate'] = 'true'

    shape_str = '_'.join(map(str, volume.shape[::-1]))
    offset_str = '_'.join(map(str, offset_zyx[::-1]))

    r = session.post(f'{server}/api/node/{uuid}/{instance}/raw/0_1_2/{shape_str}/{offset_str}',
                    params=params, data=bytes(volume))
    r.raise_for_status()


@dvid_api_wrapper
def fetch_subvolblocks(server, uuid, instance, box_zyx, compression='jpeg', throttle=False, *, session=None):
    """
    Fetch a grayscale volume using the /subvolblocks endpoint,
    which returns blocks in their compressed form.
    The blocks will be decompressed here and assembled into
    a single volume of the requested shape.

    Args:
        server:
            dvid server, e.g. 'emdata3:8900'

        uuid:
            dvid uuid, e.g. 'abc9'

        instance:
            dvid instance name, e.g. 'grayscale'

        box_zyx:
            The bounds of the volume to fetch in the coordinate system for the requested scale.
            Given as a pair of coordinates (start, stop), e.g. [(0,0,0), (10,20,30)], in Z,Y,X order.

            The box need not be block-aligned.  If it is not aligned, excess data will be requested to
            ensure only block-aligned requests, and then that excess data will be cropped out before
            the result is returned to you.
    Returns:
        np.ndarray, uint8, with shape = (box_zyx[1] - box_zyx[0])
    """
    box_zyx = np.asarray(box_zyx)
    assert np.issubdtype(box_zyx.dtype, np.integer), \
        f"Box has the wrong dtype.  Use an integer type, not {box_zyx.dtype}"
    assert box_zyx.shape == (2,3)
    assert compression == 'jpeg', "For now, only jpeg compression is supported."

    # pre-align the box
    orig_box = box_zyx
    box_zyx = round_box(box_zyx, 64, 'out')

    params = {}
    if throttle:
        params['throttle'] = 'true'

    # jpeg is default compression
    if compression != 'jpeg':
        params['compression'] = compression

    shape_zyx = (box_zyx[1] - box_zyx[0])
    shape_str = '_'.join(map(str, shape_zyx[::-1]))
    offset_str = '_'.join(map(str, box_zyx[0, ::-1]))

    r = session.get(f'{server}/api/node/{uuid}/{instance}/subvolblocks/{shape_str}/{offset_str}', params=params)
    r.raise_for_status()

    from PIL import Image

    pos = 0
    result = np.zeros(box_zyx[1] - box_zyx[0], np.uint8)
    while pos < len(r.content):
        if len(r.content) < pos+16:
            raise RuntimeError("Response from /subvolblocks is malformed: buffer truncated")
        header = r.content[pos:(pos+16)]
        bx, by, bz, nbytes = np.frombuffer(header, np.int32)
        pos += 16

        block_offset = 64*np.array([bz, by, bx])
        block_box = [block_offset, block_offset+64]
        if (block_offset < box_zyx[0]).any() or (block_offset >= box_zyx[1]).any():
            raise RuntimeError(f"Response from /subvolblocks is malformed:\n"
                               f"block_offset {block_offset.tolist()} is out-of-bounds for requested subvol {box_zyx.tolist()}")

        if len(r.content) < pos+nbytes:
            raise RuntimeError("Response from /subvolblocks is malformed: buffer truncated or block header is incorrect")
        jpeg_buf = r.content[pos:(pos+nbytes)]
        pos += nbytes

        img = Image.open(BytesIO(jpeg_buf))
        block_data = np.array(img).reshape((64,64,64))

        result_block_box = block_box - box_zyx[0]
        result[box_to_slicing(*result_block_box)] = block_data

    # Return only the requested data, discarding the
    # padding we added for alignment purposes.
    internal_box = orig_box - box_zyx[0]
    return result[box_to_slicing(*internal_box)]


def fetch_subvol(server, uuid, instance, box_zyx, *, session=None, progress=True):
    """
    Call fetch_subvolblocks() repeatedly to fetch an entire volume.
    In theory, this shouldn't be necessary since /subvolblocks is supposed to return multiple blocks at once.
    But at the moment, it appears /subvolblocks can only handle one block at a time.
    """
    box_zyx = np.asarray(box_zyx)
    assert np.issubdtype(box_zyx.dtype, np.integer), \
        f"Box has the wrong dtype.  Use an integer type, not {box_zyx.dtype}"
    assert box_zyx.shape == (2,3)

    # pre-align the box
    orig_box = box_zyx
    box_zyx = round_box(box_zyx, 64, 'out')

    result = np.zeros(box_zyx[1] - box_zyx[0], np.uint8)
    for block_box in tqdm_proxy(boxes_from_grid(box_zyx, (64,64,64)), leave=False, disable=not progress):
        res_box = block_box - box_zyx[0]
        result[box_to_slicing(*res_box)] = fetch_subvolblocks(server, uuid, instance, block_box, session=session)

    # Return only the requested data, discarding the
    # padding we added for alignment purposes.
    internal_box = orig_box - box_zyx[0]
    return result[box_to_slicing(*internal_box)]


@dvid_api_wrapper
def fetch_volume_box(server, uuid, instance, *, session=None):
    """
    Return the volume extents for the given instance as a box.
    
    Returns:
        np.ndarray [(z0,y0,x0), (z1,y1,x1)]
    
    Notes:
        - Returns *box*, shape=(box[1] - box[0])
        - Returns ZYX order
    """
    info = fetch_instance_info(server, uuid, instance, session=session)
    box_xyz = np.array((info["Extended"]["MinPoint"], info["Extended"]["MaxPoint"]))
    
    if box_xyz[0] is None or box_xyz[1] is None:
        # If the instance has been created, but not written to,
        # DVID will return null extents.
        # We return zeros, since that's nicer to work with.
        return np.array([[0,0,0], [0,0,0]])
    
    box_xyz[1] += 1
    
    box_zyx = box_xyz[:,::-1]
    return box_zyx


@dvid_api_wrapper
def post_resolution(server, uuid, instance, resolution, *, session=None):
    """
    Sets the resolution for the image volume.
    
    Args:
        server, uuid, instance:
            Refer to a voxels-like instance, e.g. uint8blk, labelmap, etc.
        
        resolution:
            For example: [8.0,  8.0, 8.0]
            
            Note:
                Following the python conventions used everywhere in this library,
                the resolution should be passed in ZYX order!
    """
    resolution = np.asarray(resolution).tolist()
    assert len(resolution) == 3
    r = session.post(f'{server}/api/node/{uuid}/{instance}/resolution', json=resolution[::-1])
    r.raise_for_status()


@dvid_api_wrapper
def post_extents(server, uuid, instance, box_zyx, *, session=None):
    """
    Post new volume extents for the given instance.
    
        server:
            dvid server, e.g. 'emdata3:8900'
        
        uuid:
            dvid uuid, e.g. 'abc9'
        
        instance:
            dvid instance name, e.g. 'grayscale'.
            Must be a volume type, e.g. 'uint8blk' or 'labelmap', etc.

        box_zyx:
            The new extents: [[z0,y0,x0], [z1,y1,x1]].
    """
    box_zyx = np.asarray(box_zyx)
    assert box_zyx.shape == (2,3)
    
    min_point_xyz = box_zyx[0, ::-1]
    max_point_xyz = box_zyx[1, ::-1] - 1

    extents_json = { "MinPoint": min_point_xyz.tolist(),
                     "MaxPoint": max_point_xyz.tolist() }
    
    url = f'{server}/api/node/{uuid}/{instance}/extents'
    r = session.post(url, json=extents_json)
    r.raise_for_status()


def update_extents(server, uuid, instance, minimal_extents_zyx, *, session=None):
    """
    Convenience function. (Not a direct endpoint wrapper.)
    
    Ensure that the given data instance has at least the given extents.
    Update the instance extents metadata along axes that are smaller
    than the given extents box.
    
    Args:
        server:
            dvid server, e.g. 'emdata3:8900'
        
        uuid:
            dvid uuid, e.g. 'abc9'
        
        instance:
            dvid instance name, e.g. 'grayscale'

        minimal_box_zyx:
            3D bounding box [min_zyx, max_zyx] = [(z0,y0,x0), (z1,y1,x1)].
            If provided, data extents will be at least this large (possibly larger).
            (The max extent should use python conventions, i.e. the MaxPoint + 1)
    Returns:
        True if the extents were modified, False otherwise
    """
    minimal_extents_zyx = np.array(minimal_extents_zyx, dtype=int)
    assert minimal_extents_zyx.shape == (2,3), \
        "Minimal extents must be provided as a 3D bounding box: [(z0,y0,x0), (z1,y1,x1)]"
    logger.info(f"Updating extents for {uuid}/{instance}")
    
    # Fetch original extents.
    info = fetch_instance_info(server, uuid, instance, session=session)
    
    orig_extents_xyz = np.array( [(1e9, 1e9, 1e9), (-1e9, -1e9, -1e9)], dtype=int )
    if info["Extended"]["MinPoint"] is not None:
        orig_extents_xyz[0] = info["Extended"]["MinPoint"]

    if info["Extended"]["MaxPoint"] is not None:
        orig_extents_xyz[1] = info["Extended"]["MaxPoint"]
        orig_extents_xyz[1] += 1

    minimal_extents_xyz = minimal_extents_zyx[:, ::-1].copy()
    minimal_extents_xyz[0] = np.minimum(minimal_extents_xyz[0], orig_extents_xyz[0])
    minimal_extents_xyz[1] = np.maximum(minimal_extents_xyz[1], orig_extents_xyz[1])

    if (minimal_extents_xyz != orig_extents_xyz).any():
        post_extents(server, uuid, instance, minimal_extents_xyz[:, ::-1])
        return True
    return False


def closest_approach(sv_vol, id_a, id_b, check_present=True):
    """
    Given a segmentation volume and two label IDs which it contains,
    find the two coordinates within id_a and id_b, respectively,
    which mark the two objects' closest approach, i.e. where the objects
    come closest to touching, even if they don't actually touch.
    
    Returns (coord_a, coord_b, distance)
    """
    assert id_a != 0 and id_b != 0, \
        "Can't use label 0 as an object ID in closest_approach()"
    
    assert sv_vol.dtype not in (np.uint64, np.int64, np.int32), \
        f"Volume type {sv_vol.dtype} is not convertible to uint32 without precision loss"
    
    mask_a = (sv_vol == id_a)
    mask_b = (sv_vol == id_b)

    if check_present and (not mask_a.any() or not mask_b.any()):
        # If either object is not present, there is no closest approach
        return (-1,-1,-1), (-1,-1,-1), np.inf
    
    if id_a == id_b:
        # IDs are identical.  Choose an arbitrary point.
        first_point = tuple(np.transpose(mask_a.nonzero())[0])
        return first_point, first_point, 0.0

    return closest_approach_between_masks(mask_a, mask_b)


def closest_approach_between_masks(mask_a, mask_b):
    """
    Given two non-overlapping binary masks,
    find the two coordinates within mask_a and mask_b, respectively,
    which mark the two objects' closest approach, i.e. where the objects
    come closest to touching, even if they don't actually touch.

    FIXME:
        This uses vigra's vectorDistanceTransform(), which uses a
        lot of RAM and computes the distance at all points in the mask.
        For sparse enough masks, it might be much more efficient to convert
        the masks to lists of coordinates and then use KD-trees to find
        the closest approach.
    """
    # Avoid circular import
    from dvid.util.segmentation import compute_nonzero_box

    # Wrapper function just for visibility to profilers
    def vectorDistanceTransform(mask):
        mask = mask.astype(np.uint32)
        mask = vigra.taggedView(mask, 'zyx'[-mask.ndim:])

        # vigra always returns the vectors (in the channel dimension)
        # in 'xyz' order, but we want zyx order!
        vdt = vigra.filters.vectorDistanceTransform(mask)
        vdt = vdt[..., ::-1]
        return vdt

    # Extract the minimal subvolume that captures both masks
    box_a = compute_nonzero_box(mask_a)
    box_b = compute_nonzero_box(mask_b)
    box_u = box_union(box_a, box_b)
    mask_a = extract_subvol(mask_a, box_u)
    mask_b = extract_subvol(mask_b, box_u)

    # For all voxels, find the shortest vector toward id_b
    to_b_vectors = vectorDistanceTransform(mask_b)

    # Magnitude of those vectors == distance to id_b
    to_b_distances = np.linalg.norm(to_b_vectors, axis=-1)

    # We're only interested in the voxels within id_a;
    # everything else is infinite distance
    to_b_distances[~mask_a] = np.inf

    # Find the point within id_a with the smallest vector
    point_a = np.unravel_index(np.argmin(to_b_distances), to_b_distances.shape)
    distance = to_b_distances[tuple(point_a)]

    # Its closest point id_b is indicated by the corresponding vector
    point_a = np.asarray(point_a, np.int32)
    point_b = (point_a + to_b_vectors[tuple(point_a)]).astype(np.int32)

    # Add the subvolume offset
    point_a = tuple(point_a + box_u[0])
    point_b = tuple(point_b + box_u[0])
    return (point_a, point_b, distance)


def approximate_closest_approach(vol, id_a, id_b, scale=1):
    """
    Like closest_approach(), but first downsamples the data (for speed).
    
    The returned coordinates may not be precisely what closest_approach would have returned,
    but they are still guaranteed to reside within the objects of interest.
    """
    mask_a = (vol == id_a)
    mask_b = (vol == id_b)

    if not mask_a.any() or not mask_b.any():
        return ((-1, -1, -1), (-1, -1, -1), np.inf)

    scaled_mask_a, _ = downsample_binary_3d_suppress_zero(mask_a, (2**scale))
    scaled_mask_b, _ = downsample_binary_3d_suppress_zero(mask_b, (2**scale))

    scaled_point_a, scaled_point_b, _ = closest_approach_between_masks(scaled_mask_a, scaled_mask_b)

    scaled_point_a = np.asarray(scaled_point_a)
    scaled_point_b = np.asarray(scaled_point_b)

    # Compute the full-res box that corresponds to the downsampled points
    point_box_a = np.array([scaled_point_a, 1+scaled_point_a]) * (2**scale)
    point_box_b = np.array([scaled_point_b, 1+scaled_point_b]) * (2**scale)
    
    point_box_a = box_intersection(point_box_a, [(0,0,0), vol.shape])
    point_box_b = box_intersection(point_box_b, [(0,0,0), vol.shape])

    # Select the first non-zero point in the full-res box
    point_a = np.transpose(extract_subvol(mask_a, point_box_a).nonzero())[0] + point_box_a[0]
    point_b = np.transpose(extract_subvol(mask_b, point_box_b).nonzero())[0] + point_box_b[0]

    distance = np.linalg.norm(point_b - point_a)
    return (tuple(point_a), tuple(point_b), distance)


def upsample(orig_data, upsample_factor):
    """
    Upsample the given array by duplicating every
    voxel into the corresponding upsampled voxels.
    """
    orig_shape = np.array(orig_data.shape)
    upsampled_data = np.empty( orig_shape * upsample_factor, dtype=orig_data.dtype )
    v = view_as_blocks(upsampled_data, orig_data.ndim*(upsample_factor,))
    
    slicing = (Ellipsis,) + (None,)*orig_data.ndim
    v[:] = orig_data[slicing]
    return upsampled_data


def downsample_mask(mask, factor, method='or'):
    """
    Downsample a boolean mask by the given factor.
    """
    assert method in ('or', 'and')

    mask = np.asarray(mask)
    assert mask.ndim >= 1
    if not isinstance(factor, Iterable):
        factor = mask.ndim*(factor,)

    factor = np.asarray(factor)
    assert (factor >= 1).all(), f"Non-positive downsampling factor: {factor}"
    assert not any(mask.shape % factor), \
        "mask shape must be divisible by the downsampling factor"

    if (factor == 1).all():
        return mask

    mask = np.asarray(mask, order='C')
    v = view_as_blocks(mask, (*factor,))
    last_axes = (*range(v.ndim),)[-mask.ndim:]

    if method == 'or':
        f = np.logical_or.reduce
    if method == 'and':
        f = np.logical_and.reduce

    return f(v, axis=last_axes)


def extract_labels_from_volume(points_df, volume, box_zyx=None, vol_scale=0, label_names=None, name_col=None):
    """
    Given a list of point coordinates and a label volume, assign a
    label to each point based on its position in the volume.

    Extracting values from an array in numpy is simple.
    In the simplest case, this is equivalent to:

        coords = points_df[['z', 'y', 'x']].values.transpose()
        points_df['label'] = volume[(*coords,)]

    But this function supports extra features:

    - Points outside the volume extents are handled gracefully (they remain unlabeled).
    - The volume can be offset from the origin (doesn't start at (0,0,0)).
    - The volume can be provided in downscaled form, in which case the
      given points will be downscaled before sampling is performed.
    - Both label values (ints) and label names are output, if the label names were specified.

    Args:
        points_df:
            DataFrame with at least columns ['x', 'y', 'z'].
            The points in this DataFrame should be provided at SCALE-0,
            regardless of vol_scale.
            This function appends two additional columns to the DataFrame, IN-PLACE.

        volume:
            3D ndarray of label voxels

        box_zyx:
            The (min,max) coordinates in which the volume resides in the point coordinate space.
            It is assumed that this box is provided at the same scale as vol_scale,
            (i.e. it is not necessarily given using scale-0 coordiantes).

        vol_scale:
            Specifies the scale at which volume (and box_zyx) were provided.
            The coordinates in points_df will be downscaled accordingly.

        label_names:
            Optional.  Specifies how label IDs map to label names.
            If provided, a new column 'label_name' will be appended to
            points_df in addition to the 'label' column.

            Must be either:
            - a mapping of `{ label_id: name }` (or `{ name : label_id }`),
              indicating each label ID in the output image, or
            - a list label names in which case the mapping is determined automatically
              by enumerating the labels in the given order (starting at 1).

        name_col:
            Customize the name of the column which will be used to store the extracted
            label and label names. Otherwise, the results are stored in the columns
            'label' and 'label_name'.

    Returns:
        None.  Results are appended to the points_df as new column(s).
    """
    if box_zyx is None:
        box_zyx = np.array(([0]*volume.ndim, volume.shape))

    assert ((box_zyx[1] - box_zyx[0]) == volume.shape).all() 

    assert points_df.index.duplicated().sum() == 0, \
        "This function doesn't work if the input DataFrame's index has duplicate values."

    downsampled_coords_zyx = (points_df[['z', 'y', 'x']] // (2**vol_scale)).astype(np.int32)

    # Drop everything outside the combined_box
    min_z, min_y, min_x = box_zyx[0] #@UnusedVariable
    max_z, max_y, max_x = box_zyx[1] #@UnusedVariable
    dc = downsampled_coords_zyx
    downsampled_coords_zyx = dc.loc[   (dc['z'] >= min_z) & (dc['z'] < max_z)
                                     & (dc['y'] >= min_y) & (dc['y'] < max_y)
                                     & (dc['x'] >= min_x) & (dc['x'] < max_x) ]
    del dc

    logger.info(f"Extracting labels from volume at {len(downsampled_coords_zyx)} points")
    downsampled_coords_zyx -= box_zyx[0]

    points_df.drop(columns=['label', 'label_name'], errors='ignore', inplace=True)
    points_df['label'] = volume.dtype.type(0)
    points_df.loc[downsampled_coords_zyx.index, 'label'] = volume[tuple(downsampled_coords_zyx.values.transpose())]

    # If no names were supplied, we're done.
    if label_names is None:
        return

    if isinstance(label_names, Mapping):
        # We need a mapping of label_ids -> names.
        # If the user provided the reverse mapping,
        # then flip it.
        (k,v) = next(iter(label_names.items()))
        if isinstance(k, str):
            # Reverse the mapping
            label_names = { v:k for k,v in label_names.items() }
    else:
        label_names = dict(enumerate(label_names, start=1))

    name_set = ['<unspecified>', *label_names.values()]
    default_names = ['<unspecified>']*len(points_df)
    # FIXME: More than half of the runtime of this function is spent on this line!
    #        Is there some way to speed this up?
    points_df['label_name'] = pd.Categorical( default_names,
                                              categories=name_set,
                                              ordered=False )
    for label, name in label_names.items():
        rows = points_df['label'] == label
        points_df.loc[rows, 'label_name'] = name

    if name_col:
        points_df.drop(columns=[name_col, f'{name_col}_label'], errors='ignore', inplace=True)
        points_df.rename(inplace=True, columns={'label': f'{name_col}_label', 'label_name': name_col})


def compute_merges(orig_vol, agg_vol):
    """
    Given an original volume and another volume which was generated
    exclusively from merges of the original, recover the merge decisions
    that were made.  That is, give the list of merges in the original
    volume that could reconstruct the geometry of segments in the
    agglomerated volume.
    
    Args:
        orig_vol:
            label volume, original segmentation

        agg_vol:
            label volume, agglomerated segmentation
    
    Returns:
        dict: { agg_id: [orig_id, orig_id, ...] },
        where the original IDs present in each merge are listed from largest to smallest.
        Agglomerated segments that exactly match an original segment (no merges) are not
        included in the results. (All lists in the results have at least two items.)
    
    Notes:
      - This function does not make any attempt to handle splits gracefully.
        For correct results, the every segment in the original volume should
        be a subset of only one segment in the agglomerated volume.
    
      - The label IDs in the agglomerated volume need not be related
        in any way to the label IDs in the original.
    """
    # Compute the set of unique orig-agg pairs, and the size of each
    df = pd.DataFrame({'orig': orig_vol.reshape(-1), 'agg': agg_vol.reshape(-1)})
    paired_seg_voxels = df.groupby(['orig', 'agg']).size().rename('voxels')
    paired_seg_voxels = pd.DataFrame(paired_seg_voxels)

    # For each agg ID with more than one corresponding 'orig' ID,
    # Compute the list of merges that reconstruct the agg geometry
    merges = {}    
    for agg, g_df in paired_seg_voxels.groupby('agg'):
        if len(g_df) > 1:
            merged_orig = g_df.sort_values('voxels', ascending=False).index.get_level_values('orig')
            merges[agg] = merged_orig.tolist()

    return merges


def unordered_duplicated(df, subset=None, keep='first'):
    """
    Like pd.DataFrame.duplicated(), but sorts each row first, so
    rows can be considered duplicates even if their values don't
    appear in the same order.

    Example:
    
        >>> df = pd.DataFrame( [(1, 2, 0.0),
                                (2, 1, 0.1), # <-- duplicate a/b columns
                                (3, 4, 0.2)],
                              columns=['a', 'b', 'score'])

        >>> unordered_duplicated(df, ['a', 'b'])
        0    False
        1     True
        2    False
        dtype: bool    
    """
    if subset is None:
        subset = list(df.columns)
    normalized_cols = np.sort(df[subset].values, axis=1)
    dupes = pd.DataFrame(normalized_cols).duplicated(keep=keep).values
    return pd.Series(dupes, index=df.index)


def drop_unordered_duplicates(df, subset=None, keep='first'):
    """
    Like pd.DataFrame.drop_duplicates(), but sorts each row first, so
    rows can be considered duplicates even if their values don't
    appear in the same order.

    Example:

        >>> df = pd.DataFrame( [(1, 2, 0.0),
                                (2, 1, 0.1), # <-- duplicate a/b columns
                                (3, 4, 0.2)],
                              columns=['a', 'b', 'score'])

        >>> drop_unordered_duplicates(df, ['a', 'b'])
           a  b  score
        0  1  2    0.0
        2  3  4    0.2

    """
    dupes = unordered_duplicated(df, subset, keep)
    return df.loc[~dupes]


def fix_df_names(df):
    """
    Rename all columns of the given dataframe with programmer-friendly alternatives,
    i.e. lowercase and replace spaces with underscores.
    """
    return df.rename(columns={c: c.lower().replace(' ', '_').replace('-', '_') for c in df.columns})


def downgrade_nullable_dtypes(df, null_handling='error'):
    """
    Convert all nullable integer columns to use a standard numpy dtype.

    New versions of pandas introduced the concept of nullable integer types,
    which are not identical to numpy dtypes:

        https://pandas.pydata.org/docs/user_guide/integer_na.html

    This function will just convert them back to regular numpy dtypes.
    If a column contains null values, the behavior depends on null_handling parameter.

    Args:
        df:
            DataFrame, possibly containing column dtypes like 'Int64' (instead of 'int64')

        null_handling:
            Either 'error', 'skip', or 'makefloat'.

            If 'error':
                Attempt to convert ALL nullable columns, and let pandas raise an exception
                if one of your columns contains a null value and therefore can't be
                downgraded to a numpy integer dtype.

            If 'skip'
                Downgrade columns which don't contain null values, but skip the ones that do.

            If 'makefloat':
                Downgrade columns which contain nulls to np.float64, as pandas uses by default.
    Returns:
        DataFrame
    """
    assert null_handling in ('error', 'skip', 'makefloat')
    pd_to_np = {'boolean': 'bool'}
    for sign, size in product(['U', ''], [8, 16, 32, 64]):
        k = f'{sign}Int{size}'
        pd_to_np[k] = k.lower()

    new_dtypes = {}
    for col, dtype in df.dtypes.items():
        if null_handling == 'skip' and df[col].isnull().any():
            continue
        elif null_handling == 'makefloat' and str(dtype) in pd_to_np and df[col].isnull().any():
            if str(dtype) == "boolean":
                new_dtypes[col] = object
            else:
                new_dtypes[col] = np.float64
        else:
            new_dtypes[col] = pd_to_np.get(str(dtype), dtype)

    return df.astype(new_dtypes)


def swap_df_cols(df, prefixes=None, swap_rows=None, suffixes=['_a', '_b']):
    """
    Swap selected columns of a dataframe, specified as a list of prefixes and two suffixes.
    Operates IN-PLACE, but incurs a full copy internally of the selected columns.

    Args:
        df:
            Input dataframe, with columns to be swapped.

        prefixes:
            columns to swap, minus their suffixes.
            If not provided, all columns with corresponding suffixes will be swapped.

        swap_rows:
            Optional.
            Specify a subset of rows in the dataframe to apply the swap to.
            Should be a Series boolean values, or a list of index values. 
            If this is a Series, it must have the same index as the input dataframe.
            If not provided, all rows are swapped.

        suffixes:
            Used to identify the left/right columns of each swapped pair.

    Returns:
        None.  Operates IN-PLACE.

    Example:
        >>> df = pd.DataFrame(np.arange(12).reshape(3,4), columns=['x_a', 'x_b', 'y_a', 'y_b'])

        >>> df
           x_a  x_b  y_a  y_b
        0    0    1    2    3
        1    4    5    6    7
        2    8    9   10   11

        >>> swap_df_cols(df, None, [True, False, True])
           x_a  x_b  y_a  y_b
        0    1    0    3    2
        1    4    5    6    7
        2    9    8   11   10

    """
    suffixes = list(suffixes)
    assert len(suffixes) == 2

    if prefixes is None:
        prefixes = set()
        suffix_len = len(suffixes[0])
        assert suffix_len == len(suffixes[1]), "Suffixes are not the same length"
        for col in df.columns:
            prefix = col[:-suffix_len]
            if (prefix + suffixes[0] in df) and (prefix + suffixes[1] in df):
                prefixes.add(prefix)
        assert prefixes, "Could not find any column pairs with the given suffixes"

    if swap_rows is None:
        swap_rows = slice(None)
    else:
        assert swap_rows.dtype == np.bool

    all_cols = [p + s for p,s in product(prefixes, suffixes)]
    missing_cols = set(all_cols) - set(df.columns)
    assert not missing_cols, \
        f"The following columns do not exist in the input DataFrame: {list(missing_cols)}"

    orig_df = df[all_cols].copy()

    for prefix in prefixes:
        col_a = prefix + suffixes[0]
        col_b = prefix + suffixes[1]
        df.loc[swap_rows, col_a] = orig_df.loc[swap_rows, col_b]
        df.loc[swap_rows, col_b] = orig_df.loc[swap_rows, col_a]