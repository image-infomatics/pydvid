
import numpy as np
from numba import jit


@jit(nopython=True, nogil=True)
def encode_coords_to_uint64(coords):
    """
    Encode an array of (N,3) int32 into an array of (N,) uint64,
    giving 21 bits per coord.
    """
    assert coords.shape[1] == 3

    N = len(coords)
    encoded_coords = np.empty(N, np.uint64)

    for i in range(N):
        z, y, x = coords[i].astype(np.int64)

        # Mask to just 21 bits
        z &= ((1 << 21) - 1)
        y &= ((1 << 21) - 1)
        x &= ((1 << 21) - 1)

        encoded = 0
        encoded |= z << 42
        encoded |= y << 21
        encoded |= x
        encoded_coords[i] = encoded

    return encoded_coords


@jit(nopython=True, nogil=True)
def decode_coords_from_uint64(encoded_coords, signed=True):
    """
    The reciprocal to encoded_coords_to_uint64(), above.

    Args:
        encoded_coords:
            A 1D array of uint64
        signed:
            If True, then interprets each encoded Z,Y,X value as a 21-bit
            signed 2's complement value, and emit negative int32 values as needed.
    """
    N = len(encoded_coords)
    coords = np.empty((N,3), np.int32)

    for i in range(N):
        encoded = encoded_coords[i]
        z = np.int32((encoded >> 2*21) & 0x1F_FFFF)  # 21 bits
        y = np.int32((encoded >>   21) & 0x1F_FFFF)  # 21 bits
        x = np.int32((encoded >>    0) & 0x1F_FFFF)  # 21 bits

        if signed:
            # Check sign bits and extend if necessary
            if encoded & (1 << (3*21-1)):
                z |= np.int32(0xFFFF_FFFF << 21)

            if encoded & (1 << (21*2-1)):
                y |= np.int32(0xFFFF_FFFF << 21)

            if encoded & (1 << (21*1-1)):
                x |= np.int32(0xFFFF_FFFF << 21)

        coords[i] = (z,y,x)

    return coords


@jit(nopython=True, nogil=True)
def encode_coords_to_blockmajor_uint64(coords, signed=True):
    """
    Encode an array of (N,3) int32 into an array of (N,) uint64,
    but arrange the bit representations such that the block ID (Bz, By, Bx)
    occupies the most significant bits and the within-block point ID (bz, by, bx)
    occupies the least significant bits.

    The block shape is hard-coded as (64,64,64), and therefore 6 bits are
    allocated to each dimension of the within-block point ID.
    The bit fields of the encoding are shown below.

    0 Bz (15 bits)    By (15 bits)    Bx (15 bits)    bz (6) by (6) bx (6)
    - --------------- --------------- --------------- ------ ------ ------
    3 210987654321098 765432109876543 210987654321098 765432 109876 543210

    If the resulting integers were then sorted, they'd appear in
    block scan-order, and voxelwise scan-order within each block.

    Args:
        coords:
            ndarray of int32, shape (N,3)
            All values must be be in range(-2**20, 2**20), unless signed=False,
            in which case the acceptable range is (0, 2**21).

        signed:
            To properly preserve sort order of negative coordinates, the block IDs
            are stored in 'offset binary' form unless signed=False,
            in which case the block IDs are stored without modification.
            See above for the acceptable coordinate ranges in the signed and
            unsigned cases.
    Returns:
        1D array, uint64, length N
    """
    assert coords.shape[1] == 3

    N = len(coords)
    encoded_coords = np.empty(N, np.uint64)

    for i in range(N):
        z, y, x = coords[i].astype(np.int64)

        # Mask to just 21 bits
        z &= ((1 << 21) - 1)
        y &= ((1 << 21) - 1)
        x &= ((1 << 21) - 1)

        if signed:
            # Convert to 'offset binary', i.e. invert the MSB
            z ^= (1 << 20)
            y ^= (1 << 20)
            x ^= (1 << 20)

        # Blocks are 64px (2**6)
        Bz = z >> 6
        By = y >> 6
        Bx = x >> 6

        mask = (1 << 6) - 1
        bz = z & mask
        by = y & mask
        bx = x & mask

        encoded = np.int64(0)
        encoded |= Bz << (15*2 + 6*3)
        encoded |= By << (15*1 + 6*3)
        encoded |= Bx << (15*0 + 6*3)
        encoded |= bz << (6*2)
        encoded |= by << (6*1)
        encoded |= bx << (6*0)
        encoded_coords[i] = np.uint64(encoded)

    return encoded_coords


@jit(nopython=True, nogil=True)
def decode_coords_from_blockmajor_uint64(encoded_coords, signed=True):
    """
    Reciprocal of encode_coords_to_blockmajor_uint64().
    Not needed very often, but useful for testing.
    """
    N = len(encoded_coords)
    coords = np.empty((N,3), np.int32)

    for i in range(N):
        encoded = encoded_coords[i]
        Bz = encoded >> (15*2 + 6*3) & ((1 << 15) - 1)
        By = encoded >> (15*1 + 6*3) & ((1 << 15) - 1)
        Bx = encoded >> (15*0 + 6*3) & ((1 << 15) - 1)

        bz = encoded >> (6*2) & ((1 << 6) - 1)
        by = encoded >> (6*1) & ((1 << 6) - 1)
        bx = encoded >> (6*0) & ((1 << 6) - 1)

        z = np.int32((Bz << 6) | bz)
        y = np.int32((By << 6) | by)
        x = np.int32((Bx << 6) | bx)

        if signed:
            # Convert from 'offset binary', i.e. invert the MSB
            z ^= (1 << 20)
            y ^= (1 << 20)
            x ^= (1 << 20)

            # Check sign bits and extend if necessary
            if z & (1 << 20):
                z |= np.int32(0xFFFF_FFFF << 21)

            if y & (1 << 20):
                y |= np.int32(0xFFFF_FFFF << 21)

            if x & (1 << 20):
                x |= np.int32(0xFFFF_FFFF << 21)

        coords[i] = (z,y,x)

    return coords


def sort_blockmajor(df, inplace=False, ignore_index=False, show_blockmajor_id=False):
    """
    Sort the given dataframe with block coordinates (assuming 64px blocks)
    and then voxel coordinate within each block.

    (This function works even when some coordinates are negative.)

    Args:
        df:
            DataFrame containing at least columns 'zyx'

        inplace:
            If True, sort in place. Otherwise return a new DataFrame.

        show_blockmajor_id:
            If True, leave the 'blockmajor_id' column in the result,
            which is the value that was used to sort the data.
            Otherwise, delete that column before returning.

    Return:
        If inplace=False, return a new DataFrame.
        Otherwise, return None.
    """
    assert set(df.columns) >= {*'zyx'}, "DataFrame must contain zyx columns"
    if not inplace:
        df = df.copy()

    # TODO: Verify acceptable coordinate min/max for 'signed blockmajor' encoding,
    #       and possibly fall back to ordinary multi-column df.sort_values()
    #       if necessary.

    df['_blockmajor_id'] = encode_coords_to_blockmajor_uint64(df[[*'zyx']].values)
    df.sort_values('_blockmajor_id', inplace=True, ignore_index=ignore_index)

    if show_blockmajor_id:
        df = df.rename(columns={'_blockmajor_id': 'blockmajor_id'})
    else:
        del df['_blockmajor_id']

    if not inplace:
        return df
