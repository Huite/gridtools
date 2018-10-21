# http://stackoverflow.com/questions/7075082/what-is-future-in-python-used-for-and-how-when-to-use-it-and-how-it-works
from __future__ import division

import numpy as np
from numba import jit

#: Interpolation method for upsampling: Take nearest source grid cell, even if it is invalid.
US_NEAREST = 10
#: Interpolation method for upsampling: Bi-linear interpolation between the 4 nearest source grid cells.
US_LINEAR = 11
US_LINEAR_MIDPOINT = 12

#: Aggregation method for downsampling: Take first valid source grid cell, ignore contribution areas.
DS_FIRST = 50
#: Aggregation method for downsampling: Take last valid source grid cell, ignore contribution areas.
DS_LAST = 51
# DS_MIN = 52
# DS_MAX = 53
#: Aggregation method for downsampling: Compute average of all valid source grid cells,
#: with weights given by contribution area.
DS_MEAN = 54
# DS_MEDIAN = 55
#: Aggregation method for downsampling: Compute most frequently seen valid source grid cell,
#: with frequency given by contribution area. Note that this mode can use an additional keyword argument
#: *mode_rank* which can be used to generate the n-th mode. See :py:function:`downsample_2d`.
DS_MODE = 56
#: Aggregation method for downsampling: Compute the biased weighted estimator of variance
#: (see https://en.wikipedia.org/wiki/Mean_square_weighted_deviation), with weights given by contribution area.
DS_VAR = 57
#: Aggregation method for downsampling: Compute the corresponding standard deviation to the biased weighted estimator
#: of variance
#: (see https://en.wikipedia.org/wiki/Mean_square_weighted_deviation), with weights given by contribution area.
DS_STD = 58

#: Constant indicating an empty 2-D mask
_NOMASK2D = np.ma.getmaskarray(np.ma.array([[0]], mask=[[0]]))

_EPS = 1e-10


def resample_2d(src, w, h, ds_method=DS_MEAN, us_method=US_LINEAR, fill_value=None, mode_rank=1, out=None):
    """
    Resample a 2-D grid to a new resolution.

    :param src: 2-D *ndarray*
    :param w: *int*
        New grid width
    :param h:  *int*
        New grid height
    :param ds_method: one of the *DS_* constants, optional
        Grid cell aggregation method for a possible downsampling
    :param us_method: one of the *US_* constants, optional
        Grid cell interpolation method for a possible upsampling
    :param fill_value: *scalar*, optional
        If ``None``, it is taken from **src** if it is a masked array,
        otherwise from *out* if it is a masked array,
        otherwise numpy's default value is used.
    :param mode_rank: *scalar*, optional
        The rank of the frequency determined by the *ds_method* ``DS_MODE``. One (the default) means
        most frequent value, zwo means second most frequent value, and so forth.
    :param out: 2-D *ndarray*, optional
        Alternate output array in which to place the result. The default is *None*; if provided, it must have the same
        shape as the expected output.
    :return: An resampled version of the *src* array.
    """
    out = _get_out(out, src, (h, w))
    if out is None:
        return src
    mask, use_mask = _get_mask(src)
    fill_value = _get_fill_value(fill_value, src, out)
    return _mask_or_not(_resample_2d(src, mask, use_mask, ds_method, us_method, fill_value, mode_rank, out),
                        src, fill_value)


def upsample_2d(src, w, h, method=US_LINEAR, fill_value=None, out=None, src_transform=None, out_transform=None):
    """
    Upsample a 2-D grid to a higher resolution by interpolating original grid cells.

    :param src: 2-D *ndarray*
    :param w: *int*
        Grid width, which must be greater than or equal to *src.shape[-1]*
    :param h:  *int*
        Grid height, which must be greater than or equal to *src.shape[-2]*
    :param method: one of the *US_* constants, optional
        Grid cell interpolation method
    :param fill_value: *scalar*, optional
        If ``None``, it is taken from **src** if it is a masked array,
        otherwise from *out* if it is a masked array,
        otherwise numpy's default value is used.
    :param out: 2-D *ndarray*, optional
        Alternate output array in which to place the result. The default is *None*; if provided, it must have the same
        shape as the expected output.
    :param src_transform: *affine* transform, optional
        Affine object containing (width of pixel, row rotation, upper left x-coordinate, column rotation, height of 
        pixel, upper left y-coordinate) of the src array. Column and row rotation are not supported.
    :param out_transform: *affine* transform, optional
        Affine object containing (width of pixel, row rotation, upper left x-coordinate, column rotation, height of 
        pixel, upper left y-coordinate) of the out array. Column and row rotation are not supported.
    :return: An upsampled version of the *src* array.
    """
    out = _get_out(out, src, (h, w))
    
    if (src_transform is None) ^ (out_transform is None):
        raise ValueError("Either no transform should be given, or both")
    elif src_transform is not None and out_transform is not None:
        src_transform, out_transform = _check_transform(src_transform, out_transform, src, out)
        src_dx, src_dy = src_transform[0], src_transform[4]
        out_dx, out_dy = out_transform[0], out_transform[4]
        if abs(out_dx) > abs(src_dx) or abs(out_dy) > abs(src_dy):
            raise ValueError("Invalid cellsize in 'out_transform'")
        use_transform = True
    else:
        src_transform = np.zeros(6,)
        out_transform = np.zeros(6,)
        use_transform = False
    
    if out is None:
        return src
    mask, use_mask = _get_mask(src)
    fill_value = _get_fill_value(fill_value, src, out)
    return _mask_or_not(_upsample_2d(src, mask, use_mask, method, fill_value, out, use_transform, src_transform, out_transform), src, fill_value)


def downsample_2d(src, w, h, method=DS_MEAN, fill_value=None, mode_rank=1, out=None, src_transform=None, out_transform=None):
    """
    Downsample a 2-D grid to a lower resolution by aggregating original grid cells.

    :param src: 2-D *ndarray*
    :param w: *int*
        Grid width, which must be less than or equal to *src.shape[-1]*
    :param h:  *int*
        Grid height, which must be less than or equal to *src.shape[-2]*
    :param method: one of the *DS_* constants, optional
        Grid cell aggregation method
    :param fill_value: *scalar*, optional
        If ``None``, it is taken from **src** if it is a masked array,
        otherwise from *out* if it is a masked array,
        otherwise numpy's default value is used.
    :param mode_rank: *scalar*, optional
        The rank of the frequency determined by the *method* ``DS_MODE``. One (the default) means
        most frequent value, zwo means second most frequent value, and so forth.
    :param out: 2-D *ndarray*, optional
        Alternate output array in which to place the result. The default is *None*; if provided, it must have the same
        shape as the expected output.
    :param src_transform: *affine* transform, optional
        Affine object/tuple/array/list containing (width of pixel, row rotation, upper left x-coordinate,
        column rotation, height of pixel, upper left y-coordinate) of the src array.
        Column and row rotation are not supported.
    :param out_transform: *affine* transform, optional
        Affine object/tuple/array/list containing (width of pixel, row rotation, upper left x-coordinate,
        column rotation, height of pixel, upper left y-coordinate) of the out array.
        Column and row rotation are not supported.
    :return: A downsampled version of the *src* array.
    """
    if method == DS_MODE and mode_rank < 1:
        raise ValueError('mode_rank must be >= 1')
    out = _get_out(out, src, (h, w))

    if (src_transform is None) ^ (out_transform is None):
        raise ValueError("Either no transform should be given, or both")
    elif src_transform is not None and out_transform is not None:
        src_transform, out_transform = _check_transform(src_transform, out_transform, src, out)
        src_dx, src_dy = src_transform[0], src_transform[4]
        out_dx, out_dy = out_transform[0], out_transform[4]
        if abs(out_dx) < abs(src_dx) or abs(out_dy) < abs(src_dy):
            raise ValueError("Invalid cellsize in 'out_transform'")
        use_transform = True
    else:
        src_transform = np.zeros(6,)
        out_transform = np.zeros(6,)
        use_transform = False

    if out is None:
        return src
    mask, use_mask = _get_mask(src)
    fill_value = _get_fill_value(fill_value, src, out)
    return _mask_or_not(_downsample_2d(src, mask, use_mask, method, fill_value, mode_rank, out, use_transform, src_transform, out_transform), src, fill_value)


def _get_out(out, src, shape):
    if out is None:
        return np.zeros(shape, dtype=src.dtype)
    else:
        if out.shape != shape:
            raise ValueError("'shape' and 'out' are incompatible")
        if out.shape == src.shape:
            return None
        return out


def _check_transform(src_transform, out_transform, src, out):
    src_transform = np.array(src_transform[:6], dtype=np.float64)
    out_transform = np.array(out_transform[:6], dtype=np.float64)
    src_dx, src_xrot, src_xcov0, src_yrot, src_dy, src_ycov0 = src_transform
    out_dx, out_xrot, out_xcov0, out_yrot, out_dy, out_ycov0 = out_transform
    src_w = src.shape[-1]
    src_h = src.shape[-2]
    out_w = out.shape[-1]
    out_h = out.shape[-2]
    src_xcov1 = src_xcov0 + src_w * src_dx
    out_xcov1 = out_xcov0 + out_w * out_dx
    src_ycov1 = src_ycov0 + src_h * src_dy
    out_ycov1 = out_ycov0 + out_h * out_dy
    if (src_xrot or src_yrot or out_xrot or out_yrot) != 0:
        raise NotImplementedError("Resampling rotated grids is not supported")
    if src_dx * out_dx < 0:
        raise ValueError("Incompatible 'src_transform' and 'out_transform': pixel widths must have the same sign")
    if src_dy * out_dy < 0:
        raise ValueError("Incompatible 'src_transform' and 'out_transform': pixel heights must have the same sign")
    if (out_xcov1 - src_xcov1) * src_dx > 0 or (out_xcov0 - src_xcov0) * src_dx < 0:
        raise ValueError("Invalid target coverage: target x outside of src")
    if (out_ycov1 - src_ycov1) * src_dy > 0 or (out_ycov0 - src_ycov0) * src_dy < 0:
        raise ValueError("Invalid target coverage: target y outside of src")
    return src_transform, out_transform


def _get_mask(src):
    if isinstance(src, np.ma.MaskedArray):
        mask = np.ma.getmask(src)
        if mask is not np.ma.nomask:
            return mask, True
    return _NOMASK2D, False


def _mask_or_not(out, src, fill_value):
    if isinstance(src, np.ma.MaskedArray):
        if not isinstance(out, np.ma.MaskedArray):
            if np.isfinite(fill_value):
                masked = np.ma.masked_equal(out, fill_value, copy=False)
            else:
                masked = np.ma.masked_invalid(out, copy=False)
            masked.set_fill_value(fill_value)
            return masked
    return out


def _get_fill_value(fill_value, src, out):
    if fill_value is None:
        if isinstance(src, np.ma.MaskedArray):
            fill_value = src.fill_value
        elif isinstance(out, np.ma.MaskedArray):
            fill_value = out.fill_value
        else:
            # use numpy's default fill_value
            fill_value = np.ma.array([0], mask=[False], dtype=src.dtype).fill_value
    return fill_value


# This function will be JIT-compiled by Numba with nopython=True,
# therefore all arg types must be either primitive scalars or numpy arrays.
# Key-value args are not allowed.
#
@jit(nopython=True)
def _resample_2d(src, mask, use_mask, ds_method, us_method, fill_value, mode_rank, out):
    src_w = src.shape[-1]
    src_h = src.shape[-2]
    out_w = out.shape[-1]
    out_h = out.shape[-2]

    if out_w < src_w and out_h < src_h:
        return _downsample_2d(src, mask, use_mask, ds_method, fill_value, mode_rank, out)
    elif out_w < src_w:
        if out_h > src_h:
            temp = np.zeros((src_h, out_w), dtype=src.dtype)
            temp = _downsample_2d(src, mask, use_mask, ds_method, fill_value, mode_rank, temp)
            # todo - write test & fix: must use mask=np.ma.getmaskarray(temp) here if use_mask==True
            return _upsample_2d(temp, mask, use_mask, us_method, fill_value, out)
        else:
            return _downsample_2d(src, mask, use_mask, ds_method, fill_value, mode_rank, out)
    elif out_h < src_h:
        if out_w > src_w:
            temp = np.zeros((out_h, src_w), dtype=src.dtype)
            temp = _downsample_2d(src, mask, use_mask, ds_method, fill_value, mode_rank, temp)
            # todo - write test & fix: must use mask=np.ma.getmaskarray(temp) here if use_mask==True
            return _upsample_2d(temp, mask, use_mask, us_method, fill_value, out)
        else:
            return _downsample_2d(src, mask, use_mask, ds_method, fill_value, mode_rank, out)
    elif out_w > src_w or out_h > src_h:
        return _upsample_2d(src, mask, use_mask, us_method, fill_value, out)
    return src


# This function will be JIT-compiled by Numba with nopython=True,
# therefore all arg types must be either primitive scalars or numpy arrays.
# Key-value args are not allowed.
#
@jit(nopython=True)
def _upsample_2d(src, mask, use_mask, method, fill_value, out, use_transform=False, src_transform=np.zeros((6,)), out_transform=np.zeros((6,))):
    src_w = src.shape[-1]
    src_h = src.shape[-2]
    out_w = out.shape[-1]
    out_h = out.shape[-2]

    if use_transform:
        src_dx, _, src_xcov0, _, src_dy, src_ycov0 = src_transform
        out_dx, _, out_xcov0, _, out_dy, out_ycov0 = out_transform
        scale_x = out_dx / src_dx
        scale_y = out_dy / src_dy
        x_offset = (out_xcov0 - src_xcov0) / src_dx
        y_offset = (out_ycov0 - src_ycov0) / src_dy
    else:
        if src_w == out_w and src_h == out_h:
            return src
        if out_w < src_w or out_h < src_h:
            raise ValueError("invalid target size")
        x_offset = 0.0
        y_offset = 0.0

    if method == US_NEAREST:
        if not use_transform:
            scale_x = src_w / out_w
            scale_y = src_h / out_h
        for out_y in range(out_h):
            src_y = int(y_offset + scale_y * out_y)
            for out_x in range(out_w):
                src_x = int(x_offset + scale_x * out_x)
                value = src[src_y, src_x]
                if np.isfinite(value) and not (use_mask and mask[src_y, src_x]):
                    out[out_y, out_x] = value
                else:
                    out[out_y, out_x] = fill_value

    elif method == US_LINEAR:
        if not use_transform:
            scale_x = (src_w - 1.0) / ((out_w - 1.0) if out_w > 1 else 1.0)
            scale_y = (src_h - 1.0) / ((out_h - 1.0) if out_h > 1 else 1.0)
        for out_y in range(out_h):
            src_yf = y_offset + scale_y * out_y
            src_y0 = int(src_yf)
            wy = src_yf - src_y0
            src_y1 = src_y0 + 1
            if src_y1 >= src_h:
                src_y1 = src_y0
            for out_x in range(out_w):
                src_xf = x_offset + scale_x * out_x
                src_x0 = int(src_xf)
                wx = src_xf - src_x0
                src_x1 = src_x0 + 1
                if src_x1 >= src_w:
                    src_x1 = src_x0
                v00 = src[src_y0, src_x0]
                v01 = src[src_y0, src_x1]
                v10 = src[src_y1, src_x0]
                v11 = src[src_y1, src_x1]
                if use_mask:
                    v00_ok = np.isfinite(v00) and not mask[src_y0, src_x0]
                    v01_ok = np.isfinite(v01) and not mask[src_y0, src_x1]
                    v10_ok = np.isfinite(v10) and not mask[src_y1, src_x0]
                    v11_ok = np.isfinite(v11) and not mask[src_y1, src_x1]
                else:
                    v00_ok = np.isfinite(v00)
                    v01_ok = np.isfinite(v01)
                    v10_ok = np.isfinite(v10)
                    v11_ok = np.isfinite(v11)
                if v00_ok and v01_ok and v10_ok and v11_ok:
                    ok = True
                    v0 = v00 + wx * (v01 - v00)
                    v1 = v10 + wx * (v11 - v10)
                    value = v0 + wy * (v1 - v0)
                elif wx < 0.5:
                    # NEAREST according to weight
                    if wy < 0.5:
                        ok = v00_ok
                        value = v00
                    else:
                        ok = v10_ok
                        value = v10
                else:
                    # NEAREST according to weight
                    if wy < 0.5:
                        ok = v01_ok
                        value = v01
                    else:
                        ok = v11_ok
                        value = v11
                if ok:
                    out[out_y, out_x] = value
                else:
                    out[out_y, out_x] = fill_value

    elif method == US_LINEAR_MIDPOINT:
        if not use_transform:
            scale_x = src_w / out_w
            scale_y = src_h / out_h
        for out_y in range(out_h):
            src_yf = y_offset + scale_y * out_y
            src_y0 = int(src_yf)
            wy = (src_yf - src_y0 - 0.5) + 0.5 * scale_y
            if wy < 0:
                if src_y0 > 0:
                    wy = 1.0 + wy
                    src_y1 = src_y0
                    src_y0 -= 1
                else:
                    src_y1 = src_y0
            else:
                src_y1 = src_y0 + 1
            if src_y1 >= src_h:
                src_y1 = src_y0
            for out_x in range(out_w):
                src_xf = x_offset + scale_x * out_x
                src_x0 = int(src_xf)
                wx = (src_xf - src_x0 - 0.5) + 0.5 * scale_x 
                if wx < 0:
                    if src_x0 > 0:
                        wx = 1.0 + wx
                        src_x1 = src_x0
                        src_x0 -= 1
                    else:
                        src_x1 = src_x0
                else:
                    src_x1 = src_x0 + 1
                if src_x1 >= src_w:
                    src_x1 = src_x0
                v00 = src[src_y0, src_x0]
                v01 = src[src_y0, src_x1]
                v10 = src[src_y1, src_x0]
                v11 = src[src_y1, src_x1]
                if use_mask:
                    v00_ok = np.isfinite(v00) and not mask[src_y0, src_x0]
                    v01_ok = np.isfinite(v01) and not mask[src_y0, src_x1]
                    v10_ok = np.isfinite(v10) and not mask[src_y1, src_x0]
                    v11_ok = np.isfinite(v11) and not mask[src_y1, src_x1]
                else:
                    v00_ok = np.isfinite(v00)
                    v01_ok = np.isfinite(v01)
                    v10_ok = np.isfinite(v10)
                    v11_ok = np.isfinite(v11)
                if v00_ok and v01_ok and v10_ok and v11_ok:
                    ok = True
                    v0 = v00 + wx * (v01 - v00)
                    v1 = v10 + wx * (v11 - v10)
                    value = v0 + wy * (v1 - v0)
                elif wx < 0.5:
                    # NEAREST according to weight
                    if wy < 0.5:
                        ok = v00_ok
                        value = v00
                    else:
                        ok = v10_ok
                        value = v10
                else:
                    # NEAREST according to weight
                    if wy < 0.5:
                        ok = v01_ok
                        value = v01
                    else:
                        ok = v11_ok
                        value = v11
                if ok:
                    out[out_y, out_x] = value
                else:
                    out[out_y, out_x] = fill_value
    else:
        raise ValueError('invalid upsampling method')

    return out


# This function will be JIT-compiled by Numba with nopython=True,
# therefore all arg types must be either primitive scalars or numpy arrays.
# Key-value args are not allowed.
#
@jit(nopython=True)
def _downsample_2d(src, mask, use_mask, method, fill_value, mode_rank, out, use_transform=False, src_transform=np.zeros((6,)), out_transform=np.zeros(6,)):
    # np.zeros is as default necessary for numba, it tries to unpack src_transform and out_transform
    src_w = src.shape[-1]
    src_h = src.shape[-2]
    out_w = out.shape[-1]
    out_h = out.shape[-2]

    if use_transform:
        src_dx, _, src_xcov0, _, src_dy, src_ycov0 = src_transform
        out_dx, _, out_xcov0, _, out_dy, out_ycov0 = out_transform
        scale_x = out_dx / src_dx
        scale_y = out_dy / src_dy
        x_offset = (out_xcov0 - src_xcov0) / src_dx
        y_offset = (out_ycov0 - src_ycov0) / src_dy
    else:
        if src_w == out_w and src_h == out_h:
            return src
        if out_w > src_w or out_h > src_h:
            raise ValueError("invalid target size")
        scale_x = src_w / out_w
        scale_y = src_h / out_h
        x_offset = 0.0
        y_offset = 0.0

    if method == DS_FIRST or method == DS_LAST:
        for out_y in range(out_h):
            src_yf0 = y_offset + scale_y * out_y
            src_yf1 = src_yf0 + scale_y
            src_y0 = int(src_yf0)
            src_y1 = int(src_yf1)
            if src_y1 == src_yf1 and src_y1 > src_y0:
                src_y1 -= 1
            for out_x in range(out_w):
                src_xf0 = x_offset + scale_x * out_x
                src_xf1 = src_xf0 + scale_x
                src_x0 = int(src_xf0)
                src_x1 = int(src_xf1)
                if src_x1 == src_xf1 and src_x1 > src_x0:
                    src_x1 -= 1
                done = False
                value = fill_value
                for src_y in range(src_y0, src_y1 + 1):
                    for src_x in range(src_x0, src_x1 + 1):
                        v = src[src_y, src_x]
                        if np.isfinite(v) and not (use_mask and mask[src_y, src_x]):
                            value = v
                            if method == DS_FIRST:
                                done = True
                                break
                    if done:
                        break
                out[out_y, out_x] = value

    elif method == DS_MODE:
        max_value_count = int(scale_x + 1) * int(scale_y + 1)
        values = np.zeros((max_value_count,), dtype=src.dtype)
        frequencies = np.zeros((max_value_count,), dtype=np.uint32)
        for out_y in range(out_h):
            src_yf0 = y_offset + scale_y * out_y
            src_yf1 = src_yf0 + scale_y
            src_y0 = int(src_yf0)
            src_y1 = int(src_yf1)
            wy0 = 1.0 - (src_yf0 - src_y0)
            wy1 = src_yf1 - src_y1
            if wy1 < _EPS:
                wy1 = 1.0
                if src_y1 > src_y0:
                    src_y1 -= 1
            for out_x in range(out_w):
                src_xf0 = x_offset + scale_x * out_x
                src_xf1 = src_xf0 + scale_x
                src_x0 = int(src_xf0)
                src_x1 = int(src_xf1)
                wx0 = 1.0 - (src_xf0 - src_x0)
                wx1 = src_xf1 - src_x1
                if wx1 < _EPS:
                    wx1 = 1.0
                    if src_x1 > src_x0:
                        src_x1 -= 1
                value_count = 0
                for src_y in range(src_y0, src_y1 + 1):
                    wy = wy0 if (src_y == src_y0) else wy1 if (src_y == src_y1) else 1.0
                    for src_x in range(src_x0, src_x1 + 1):
                        wx = wx0 if (src_x == src_x0) else wx1 if (src_x == src_x1) else 1.0
                        v = src[src_y, src_x]
                        if np.isfinite(v) and not (use_mask and mask[src_y, src_x]):
                            w = wx * wy
                            found = False
                            for i in range(value_count):
                                if v == values[i]:
                                    frequencies[i] += w
                                    found = True
                                    break
                            if not found:
                                values[value_count] = v
                                frequencies[value_count] = w
                                value_count += 1
                w_max = -1.
                value = fill_value
                if mode_rank == 1:
                    for i in range(value_count):
                        w = frequencies[i]
                        if w > w_max:
                            w_max = w
                            value = values[i]
                elif mode_rank <= max_value_count:
                    max_frequencies = np.full(mode_rank, -1.0, dtype=np.float64)
                    indices = np.zeros(mode_rank, dtype=np.int64)
                    for i in range(value_count):
                        w = frequencies[i]
                        for j in range(mode_rank):
                            if w > max_frequencies[j]:
                                max_frequencies[j] = w
                                indices[j] = i
                                break
                    value = values[indices[mode_rank - 1]]

                out[out_y, out_x] = value

    elif method == DS_MEAN:
        for out_y in range(out_h):
            src_yf0 = y_offset + scale_y * out_y
            src_yf1 = src_yf0 + scale_y
            src_y0 = int(src_yf0)
            src_y1 = int(src_yf1)
            wy0 = 1.0 - (src_yf0 - src_y0)
            wy1 = src_yf1 - src_y1
            if wy1 < _EPS:
                wy1 = 1.0
                if src_y1 > src_y0:
                    src_y1 -= 1
            for out_x in range(out_w):
                src_xf0 = x_offset + scale_x * out_x
                src_xf1 = src_xf0 + scale_x
                src_x0 = int(src_xf0)
                src_x1 = int(src_xf1)
                wx0 = 1.0 - (src_xf0 - src_x0)
                wx1 = src_xf1 - src_x1
                if wx1 < _EPS:
                    wx1 = 1.0
                    if src_x1 > src_x0:
                        src_x1 -= 1
                v_sum = 0.0
                w_sum = 0.0
                for src_y in range(src_y0, src_y1 + 1):
                    wy = wy0 if (src_y == src_y0) else wy1 if (src_y == src_y1) else 1.0
                    for src_x in range(src_x0, src_x1 + 1):
                        wx = wx0 if (src_x == src_x0) else wx1 if (src_x == src_x1) else 1.0
                        v = src[src_y, src_x]
                        if np.isfinite(v) and not (use_mask and mask[src_y, src_x]):
                            w = wx * wy
                            v_sum += w * v
                            w_sum += w
                if w_sum < _EPS:
                    out[out_y, out_x] = fill_value
                else:
                    out[out_y, out_x] = v_sum / w_sum

    elif method == DS_VAR or method == DS_STD:
        for out_y in range(out_h):
            src_yf0 = y_offset + scale_y * out_y
            src_yf1 = src_yf0 + scale_y
            src_y0 = int(src_yf0)
            src_y1 = int(src_yf1)
            wy0 = 1.0 - (src_yf0 - src_y0)
            wy1 = src_yf1 - src_y1
            if wy1 < _EPS:
                wy1 = 1.0
                if src_y1 > src_y0:
                    src_y1 -= 1
            for out_x in range(out_w):
                src_xf0 = x_offset + scale_x * out_x
                src_xf1 = src_xf0 + scale_x
                src_x0 = int(src_xf0)
                src_x1 = int(src_xf1)
                wx0 = 1.0 - (src_xf0 - src_x0)
                wx1 = src_xf1 - src_x1
                if wx1 < _EPS:
                    wx1 = 1.0
                    if src_x1 > src_x0:
                        src_x1 -= 1
                v_sum = 0.0
                w_sum = 0.0
                wv_sum = 0.0
                wvv_sum = 0.0
                for src_y in range(src_y0, src_y1 + 1):
                    wy = wy0 if (src_y == src_y0) else wy1 if (src_y == src_y1) else 1.0
                    for src_x in range(src_x0, src_x1 + 1):
                        wx = wx0 if (src_x == src_x0) else wx1 if (src_x == src_x1) else 1.0
                        v = src[src_y, src_x]
                        if np.isfinite(v) and not (use_mask and mask[src_y, src_x]):
                            w = wx * wy
                            v_sum += v
                            w_sum += w
                            wv_sum += w * v
                            wvv_sum += w * v * v
                if w_sum < _EPS:
                    out[out_y, out_x] = fill_value
                else:
                    out[out_y, out_x] = (wvv_sum * w_sum - wv_sum * wv_sum) / w_sum / w_sum
        if method == DS_STD:
            out = np.sqrt(out).astype(out.dtype)
    else:
        raise ValueError('invalid upsampling method')

    return out
