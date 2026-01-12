from __future__ import annotations
import math
from typing import Literal
from functools import partial

import jax
import jax_dataclasses as jdc
import jax.numpy as jnp

from ._gaussian_splat import Gaussian2D
from ._select_hit_gs import get_intersects_per_patch

try:
    from ._rasterize_warp import _rasterize_tile_warp
except ImportError:
    _rasterize_tile_warp = None


@jdc.jit
def rasterize(
    gaussians: Gaussian2D,
    depth: jnp.ndarray,
    img_height: jdc.Static[int],
    img_width: jdc.Static[int],
    tile_size: jdc.Static[int] = 40,
    max_intersects: jdc.Static[int] = 100,
    mode: jdc.Static[Literal["jax", "warp"]] = "jax",
) -> jnp.ndarray:
    gaussians.verify_shape()

    n_tiles_along_height = math.ceil(img_height / tile_size)
    n_tiles_along_width = math.ceil(img_width / tile_size)
    img_height_padded = n_tiles_along_height * tile_size
    img_width_padded = n_tiles_along_width * tile_size
    num_tiles = n_tiles_along_height * n_tiles_along_width

    tiles = jnp.stack(
        jnp.meshgrid(
            jnp.arange(n_tiles_along_width),
            jnp.arange(n_tiles_along_height),
        ),
        axis=-1,
    ).reshape(-1, 2)
    tiles = jnp.concatenate([tiles * tile_size, (tiles + 1) * tile_size], axis=-1)

    if _rasterize_tile_warp is not None and mode == "warp":
        tile_size = max(img_height, img_width)
        rasterize_fn = _rasterize_tile_warp

        img_tiles = jax.vmap(
            lambda tile: rasterize_fn(gaussians, tile, tile_size, depth)
        )(tiles)

    elif mode == "jax":
        assert max_intersects > 0
        rasterize_fn = _rasterize_tile_jax_fn

        # Pre-compute cached values to avoid repeated exp()/sigmoid() calls
        cached_means = gaussians.means
        cached_colors = gaussians.colors  # triggers sigmoid() once
        cached_opacity = gaussians.opacity  # triggers sigmoid() once
        cached_quat_matrix = gaussians.quat.as_matrix()  # compute once
        cached_scale_sq = gaussians.scale ** 2  # triggers exp() once

        # Pre-compute bounding box once for all tiles
        cached_bbox = gaussians.get_bbox()

        hit_indices = get_intersects_per_patch(
            gaussians,
            tiles,
            depth,
            img_height_padded,
            img_width_padded,
            n_tiles_along_height,
            n_tiles_along_width,
            num_tiles,
            tile_size,
            max_intersects,
            bbox=cached_bbox,
        )

        img_tiles = jax.vmap(
            lambda tile, hit_idx: rasterize_fn(
                cached_means, cached_colors, cached_opacity,
                cached_quat_matrix, cached_scale_sq,
                tile, tile_size, hit_idx
            )
        )(tiles, hit_indices)

    else:
        raise ValueError(
            f"Incompatible mode and available rasterizers: {mode}, {_rasterize_tile_warp}"
        )

    assert img_tiles.shape == (num_tiles, tile_size, tile_size, 3)

    img = img_tiles.reshape(
        n_tiles_along_height,
        n_tiles_along_width,
        tile_size,
        tile_size,
        3,
    )
    img = img.transpose(0, 2, 1, 3, 4)
    img = img.reshape(img_height_padded, img_width_padded, 3)
    img = img[:img_height, :img_width]
    return img



def _rasterize_tile_jax_fn(
    means: jnp.ndarray,
    colors: jnp.ndarray,
    opacity: jnp.ndarray,
    quat_matrix: jnp.ndarray,
    scale_sq: jnp.ndarray,
    tile: jnp.ndarray,
    tile_size: jdc.Static[int],
    intersection: jnp.ndarray,
) -> jnp.ndarray:
    """Rasterize a single tile using fori_loop with unrolling."""
    indices = jnp.stack(
        jnp.meshgrid(
            jnp.arange(tile_size) + tile[0],
            jnp.arange(tile_size) + tile[1],
        ),
        axis=-1,
    )

    # Extract only the intersecting gaussians' data
    safe_intersection = jnp.clip(intersection, 0, means.shape[0] - 1)
    local_means = means[safe_intersection]
    local_colors = colors[safe_intersection]
    local_opacity = opacity[safe_intersection]
    local_quat_matrix = quat_matrix[safe_intersection]
    local_scale_sq = scale_sq[safe_intersection]

    # Compute inverse covariances only for intersecting gaussians (~100 instead of all)
    local_inv_covs = jnp.einsum(
        "...ij,...j,...kj->...ik",
        local_quat_matrix,
        1 / local_scale_sq,
        local_quat_matrix,
    )

    # Validity mask for intersection indices
    valid_mask = intersection >= 0

    def get_alpha_and_color(local_idx):
        mean = local_means[local_idx]
        color = local_colors[local_idx]
        opac = local_opacity[local_idx]
        diff = indices - mean[None, None, :2]
        exponent = -0.5 * jnp.einsum(
            "...j,...jk,...k->...", diff, local_inv_covs[local_idx], diff
        )
        _alpha = jnp.exp(exponent) * opac
        return _alpha, color

    # fori_loop with unroll - better than vmap+cumprod because:
    # - Constant memory (no allocation for all intermediate alphas)
    # - Implicit early termination (trans→0 means contributions→0)
    def body_fn(i, state):
        img, alphas, trans = state
        _alpha, _color = get_alpha_and_color(i)
        img = img + (
            _color[..., None, None, :]
            * _alpha[..., None]
            * trans[..., None]
            * valid_mask[i][..., None, None, None]
        )
        alphas = alphas + _alpha
        trans = trans * (1 - _alpha)
        return (img, alphas, trans)

    img, _, _ = jax.lax.fori_loop(
        0,
        intersection.shape[0],
        body_fn,
        (
            jnp.zeros((tile_size, tile_size, 3)),
            jnp.zeros((tile_size, tile_size)),
            jnp.ones((tile_size, tile_size)),
        ),
        unroll=10,
    )

    img = img.clip(0, 1)
    return img
