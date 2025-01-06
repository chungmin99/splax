"""
Determine which gaussian should be prioritized for each pixel in the rasterized image.
This is important for constant `max_intersects`.
"""

import jax
import jax_dataclasses as jdc
import jax.numpy as jnp

try:
    import warp as wp
    from warp.jax_experimental import jax_kernel
except ImportError:
    wp = None
    jax_kernel = None

from ._gaussian_splat import Gaussian2D


def get_intersects_per_patch(
    gaussians: Gaussian2D,
    tiles: jnp.ndarray,
    depth: jnp.ndarray,
    img_height_padded: jdc.Static[int],
    img_width_padded: jdc.Static[int],
    n_tiles_along_height: jdc.Static[int],
    n_tiles_along_width: jdc.Static[int],
    num_tiles: jdc.Static[int],
    tile_size: jdc.Static[int],
    max_intersects: jdc.Static[int],
) -> jnp.ndarray:
    if False and wp is not None and jax_kernel is not None:
        # Experimental; WIP heuristic for choosing the "best" gaussians to render,
        # such that there are no tile-level discontinuities even with `max_intersects`.
        # This actually fails for random init with `train_image.py`...
        intersects = _get_intersections_from_first_hit(
            gaussians,
            img_height_padded,
            img_width_padded,
            n_tiles_along_height,
            n_tiles_along_width,
            num_tiles,
            tile_size,
            max_intersects,
        )
    else:
        intersects = jax.vmap(
            lambda tile: _get_intersections_from_depth(
                gaussians,
                depth,
                tile,
                max_intersects,
            )
        )(tiles)

    assert intersects.shape == (num_tiles, max_intersects)
    return intersects


def _get_intersections_from_depth(
    g2d: Gaussian2D,
    depth: jnp.ndarray,
    tile: jnp.ndarray,
    max_intersects: jdc.Static[int],
) -> jnp.ndarray:
    bbox = g2d.get_bbox()
    in_bounds = jnp.logical_and(
        jnp.logical_and(bbox[:, 2] >= tile[0], bbox[:, 0] <= tile[2]),
        jnp.logical_and(bbox[:, 3] >= tile[1], bbox[:, 1] <= tile[3]),
    )
    in_bounds = jnp.logical_and(in_bounds, depth > 0)

    intersection = jnp.nonzero(in_bounds, size=max_intersects, fill_value=-1)[0]
    return intersection


def _get_intersections_from_first_hit(
    gaussians: Gaussian2D,
    img_height_padded: int,
    img_width_padded: int,
    n_tiles_along_height: int,
    n_tiles_along_width: int,
    num_tiles: int,
    tile_size: int,
    max_intersects: jdc.Static[int],
):
    assert wp is not None and jax_kernel is not None

    # Intersect choice is like `argsort`, should not need differentiation.
    gaussians = jax.lax.stop_gradient(gaussians)

    indices = jnp.stack(
        jnp.meshgrid(
            jnp.arange(img_width_padded),
            jnp.arange(img_height_padded),
        ),
        axis=-1,
    )
    inv_covs = jnp.einsum(
        "...ij,...j,...kj->...ik",
        gaussians.quat.as_matrix(),
        1 / (gaussians.scale**2),
        gaussians.quat.as_matrix(),
    )
    hit_indices = (
        (
            jax_kernel(_get_first_hit)(
                indices.astype(jnp.float32),
                gaussians.get_bbox(),
                gaussians.means,
                gaussians.opacity,
                inv_covs,
            )[0]
        )
        .reshape(n_tiles_along_height, tile_size, n_tiles_along_width, tile_size)
        .transpose(0, 2, 1, 3)
        .reshape(num_tiles, tile_size, tile_size)
    )
    intersection = jax.vmap(
        lambda x: jnp.unique(x, size=max_intersects, fill_value=-1).astype(jnp.int32)
    )(hit_indices)
    return intersection


if wp is not None:
    @wp.kernel
    def _get_first_hit(
        indices: wp.array2d(dtype=wp.vec2),  # (tile_size, tile_size, 2)
        bbox: wp.array(dtype=wp.vec4),
        means: wp.array(dtype=wp.vec2),
        opacity: wp.array(dtype=wp.float32),
        inv_covs: wp.array(dtype=wp.mat22f),
        *,
        hit_indices: wp.array2d(dtype=wp.int32),
    ):
        i, j = wp.tid()

        hit_indices[i, j] = -1

        for idx in range(bbox.shape[0]):
            # Bounding box culling math.
            if bbox[idx][2] < indices[i, j][0] or bbox[idx][0] > indices[i, j][0]:
                continue
            if bbox[idx][3] < indices[i, j][1] or bbox[idx][1] > indices[i, j][1]:
                continue

            hit_indices[i, j] = min(idx, hit_indices[i, j])

            inv_cov = inv_covs[idx]
            diff = indices[i, j] - means[idx]
            exponent = -0.5 * wp.dot(diff, wp.mul(inv_cov, diff))
            _alpha = wp.exp(exponent) * opacity[idx]

            if _alpha < 0.2:
                continue

            hit_indices[i, j] = idx
            break
