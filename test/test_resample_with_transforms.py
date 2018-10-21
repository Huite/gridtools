import numpy as np
import pytest
import gridtools.resampling as gtr


SRC = np.array(
    [
        [0.9, 0.5, 3.0, 4.0],
        [1.1, 1.5, 1.0, 2.0],
        [4.0, 2.1, 3.0, 5.0],
        [3.0, 4.9, 3.0, 1.0],
    ]
)

SRC_TRANSFORM = (1.0, 0.0, 0.0, 0.0, -1.0, 4.0)
OUT_TRANSFORM = (1.0, 0.0, 0.5, 0.0, -1.0, 3.5)


def test_downsample_2d_first():
    assert np.array_equal(
        gtr.downsample_2d(SRC, 2, 2, gtr.DS_FIRST, src_transform=SRC_TRANSFORM, out_transform=OUT_TRANSFORM),
        np.array([
            [0.9, 0.5],
            [1.1, 1.5],
        ])
    )


def test_downsample_2d_mean():
    assert np.array_equal(
        gtr.downsample_2d(SRC, 2, 2, gtr.DS_MEAN, src_transform=SRC_TRANSFORM, out_transform=OUT_TRANSFORM),
        np.array([
            [(0.9 + 0.5 + 1.1 + 1.5) / 4.0, (0.5 + 3.0 + 1.5 + 1.0) / 4.0],
            [(1.1 + 1.5 + 4.0 + 2.1) / 4.0, (1.5 + 1.0 + 2.1 + 3.0) / 4.0],
        ])
    )


def test_downsample_2d_mean_positive_dy():
    src_transform = (1.0, 0.0, 0.0, 0.0, 1.0, -4.0)
    out_transform = (1.0, 0.0, 0.5, 0.0, 1.0, -3.5)
    assert np.array_equal(
        gtr.downsample_2d(SRC, 2, 2, gtr.DS_MEAN, src_transform=src_transform, out_transform=out_transform),
        np.array([
            [(0.9 + 0.5 + 1.1 + 1.5) / 4.0, (0.5 + 3.0 + 1.5 + 1.0) / 4.0],
            [(1.1 + 1.5 + 4.0 + 2.1) / 4.0, (1.5 + 1.0 + 2.1 + 3.0) / 4.0],
        ])
    )


def test_upsample_2d_nearest():
    out_transform = (0.75, 0.0, 0.50, 0.0, -0.75, 3.5)
    assert np.array_equal(
        gtr.upsample_2d(SRC, 2, 2, gtr.US_NEAREST, src_transform=SRC_TRANSFORM, out_transform=out_transform),
        np.array([
            [0.9, 0.5],
            [1.1, 1.5]
        ])
    )


def test_upsample_2d_linear():
    src = np.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
        ]
    )
    out_transform = (0.5, 0.0, 0.50, 0.0, -0.5, 3.5)
    print(gtr.upsample_2d(src, 2, 2, gtr.US_LINEAR_MIDPOINT, src_transform=SRC_TRANSFORM, out_transform=out_transform))
    assert np.array_equal(
        gtr.upsample_2d(src, 2, 2, gtr.US_LINEAR_MIDPOINT, src_transform=SRC_TRANSFORM, out_transform=out_transform),
        np.array([
            [0.25, 0.25],
            [0.75, 0.75]
        ])
    )


def test_errors():
    with pytest.raises(ValueError): # only one transform
        gtr.downsample_2d(SRC, 2, 2, gtr.DS_MEAN, src_transform=SRC_TRANSFORM)
    with pytest.raises(ValueError): # only one transform
        gtr.upsample_2d(SRC, 2, 2, gtr.US_NEAREST, src_transform=SRC_TRANSFORM)
    with pytest.raises(ValueError): # source has smaller grid cells than out
        src_transform = (0.1, 0.0, 0.0, 0.0, -0.1, 4.0)
        gtr.upsample_2d(SRC, 10, 10, gtr.US_NEAREST, src_transform=src_transform, out_transform=OUT_TRANSFORM)
    with pytest.raises(ValueError): # source has larger grid cells than out
        src_transform = (2.0, 0.0, 0.0, 0.0, -2.0, 4.0)
        gtr.downsample_2d(SRC, 2, 2, gtr.DS_MEAN, src_transform=src_transform, out_transform=OUT_TRANSFORM)
    with pytest.raises(NotImplementedError): # rotated grid
        src_transform = (1.0, 45.0, 0.0, 0.0, -1.0, 4.0)
        gtr.downsample_2d(SRC, 2, 2, gtr.DS_MEAN, src_transform=src_transform, out_transform=OUT_TRANSFORM)
    with pytest.raises(ValueError): # negative pixel width, source only
        src_transform = (-1.0, 0.0, 0.0, 0.0, -1.0, 4.0)
        gtr.downsample_2d(SRC, 2, 2, gtr.DS_MEAN, src_transform=src_transform, out_transform=OUT_TRANSFORM)
    with pytest.raises(ValueError): # positive pixel width, source only
        src_transform = (1.0, 0.0, 0.0, 0.0, 1.0, 4.0)
        gtr.downsample_2d(SRC, 2, 2, gtr.DS_MEAN, src_transform=src_transform, out_transform=OUT_TRANSFORM)
    with pytest.raises(ValueError): # out x falls outside of source
        gtr.downsample_2d(SRC, 4, 2, gtr.DS_MEAN, src_transform=SRC_TRANSFORM)
    with pytest.raises(ValueError): # out y falls outside of source
        gtr.downsample_2d(SRC, 2, 4, gtr.DS_MEAN, src_transform=SRC_TRANSFORM)