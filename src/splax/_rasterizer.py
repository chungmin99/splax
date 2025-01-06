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
        )

        img_tiles = jax.vmap(
            lambda tile, hit_idx: rasterize_fn(
                gaussians, tile, tile_size, hit_idx
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
    g2d: Gaussian2D,
    tile: jnp.ndarray,
    tile_size: jdc.Static[int],
    intersection: jnp.ndarray,
) -> jnp.ndarray:
    indices = jnp.stack(
        jnp.meshgrid(
            jnp.arange(tile_size) + tile[0],
            jnp.arange(tile_size) + tile[1],
        ),
        axis=-1,
    )

    inv_covs = jnp.einsum(
        "...ij,...j,...kj->...ik",
        g2d.quat.as_matrix(),
        1 / (g2d.scale**2),
        g2d.quat.as_matrix(),
    )

    def get_alpha_and_color(curr_idx):
        mean = g2d.means[curr_idx]
        color = g2d.colors[curr_idx]
        opacity = g2d.opacity[curr_idx]
        diff = indices - mean[None, None, :2]
        exponent = -0.5 * jnp.einsum(
            "...j,...jk,...k->...", diff, inv_covs[curr_idx], diff
        )
        _alpha = jnp.exp(exponent) * opacity
        return _alpha, color

    # Rasterize all the selected gaussians in the tile.
    # We use `fori_loop` combined with unroll.
    # (We could also use `jax.lax.map` with batch_size=10.)

    # This effectively performs the comment in the inspirational PR:
    #   > After the Gaussians are sorted it seems possible to chunk them
    #   > by distance, rasterize separately, and then alpha-composite?

    # fori_loop implementation.
    def body_fn(i, state):
        img, alphas, trans = state
        _alpha, _color = get_alpha_and_color(intersection[i])
        img = img + (
            _color[..., None, None, :]
            * _alpha[..., None]
            * trans[..., None]
            * (intersection[i] >= 0)[..., None, None, None]
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

    # # For reference: vmap implementation.
    # alphas, colors = jax.vmap(get_alpha_and_color)(intersection)
    # trans = jnp.nancumprod(jnp.roll(1 - alphas, 1, axis=0).at[0].set(1.0), axis=0)
    # img = jnp.sum(
    #     colors[..., None, None, :]
    #     * alphas[..., None]
    #     * trans[..., None]
    #     * (intersection >= 0)[..., None, None, None],
    #     axis=0,
    # )

    img = img.clip(0, 1)
    return img
