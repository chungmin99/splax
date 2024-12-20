from __future__ import annotations
from typing import Literal
from functools import partial

import jax
import jax_dataclasses as jdc
import jax.numpy as jnp

from ._gaussian_splat import Gaussians
from ._camera import Camera

try:
    from ._rasterize_warp import _rasterize_tile_warp
except ImportError:
    _rasterize_tile_warp = None


@jdc.jit
def rasterize(
    camera: Camera,
    gaussians: Gaussians,
    depth: jnp.ndarray,
    tile_size: jdc.Static[int] = 50,
    max_intersects: jdc.Static[int] = 100,
    mode: jdc.Static[Literal["jax", "warp"]] = "warp",
) -> jnp.ndarray:
    assert gaussians.get_and_check_shape() == 2

    if _rasterize_tile_warp is not None and mode == "warp":
        rasterize_fn = _rasterize_tile_warp
    elif mode == "jax":
        rasterize_fn = partial(_rasterize_tile_jax, max_intersects=max_intersects)
    else:
        raise ValueError(
            f"Incompatible mode and available rasterizers: {mode}, {_rasterize_tile_warp}"
        )

    n_tiles_along_height = camera.height // tile_size
    n_tiles_along_width = camera.width // tile_size
    num_tiles = n_tiles_along_height * n_tiles_along_width

    img_tiles = rasterize_fn(gaussians, camera, depth, tile_size)
    assert img_tiles.shape == (num_tiles, tile_size, tile_size, 3)

    img = img_tiles.reshape(
        camera.height // tile_size,
        camera.width // tile_size,
        tile_size,
        tile_size,
        3,
    )
    img = img.transpose(0, 2, 1, 3, 4)
    img = img.reshape(camera.height, camera.width, 3)
    return img


def _rasterize_tile_jax(
    g2d: Gaussians,
    camera: Camera,
    depth: jnp.ndarray,
    tile_size: jdc.Static[int],
    max_intersects: jdc.Static[int],
) -> jnp.ndarray:
    assert max_intersects > 0
    tiles = jnp.stack(
        jnp.meshgrid(
            jnp.arange(camera.width // tile_size),
            jnp.arange(camera.height // tile_size),
        ),
        axis=-1,
    ).reshape(-1, 2)
    tiles = jnp.concatenate([tiles * tile_size, (tiles + 1) * tile_size], axis=-1)
    rendered_tiles = jax.vmap(
        lambda tile: _rasterize_tile_jax_fn(g2d, tile, depth, tile_size, max_intersects)
    )(tiles)
    return rendered_tiles


@staticmethod
def _rasterize_tile_jax_fn(
    g2d: Gaussians,
    tile: jnp.ndarray,
    depth: jnp.ndarray,
    tile_size: jdc.Static[int],
    max_intersects: jdc.Static[int],
) -> jnp.ndarray:
    bbox = g2d.get_bbox()
    in_bounds = jnp.logical_and(
        jnp.logical_and(bbox[:, 2] >= tile[0], bbox[:, 0] <= tile[2]),
        jnp.logical_and(bbox[:, 3] >= tile[1], bbox[:, 1] <= tile[3]),
    )
    in_bounds = jnp.logical_and(in_bounds, depth > 0)

    indices = jnp.stack(
        jnp.meshgrid(
            jnp.arange(tile_size) + tile[0],
            jnp.arange(tile_size) + tile[1],
        ),
        axis=-1,
    )

    intersection = jnp.nonzero(in_bounds, size=max_intersects, fill_value=-1)[0]
    inv_covs = jnp.einsum(
        "...ij,...j,...kj->...ik",
        g2d.quat.as_matrix(),
        1 / (g2d.scale**2),
        g2d.quat.as_matrix(),
    )

    def get_alpha_and_color(idx):
        curr_idx = intersection[idx]
        mean = g2d.means[curr_idx]
        color = g2d.colors[curr_idx]
        opacity = g2d.opacity[curr_idx]
        diff = indices - mean[None, None, :2]
        exponent = -0.5 * jnp.einsum(
            "...j,...jk,...k->...", diff, inv_covs[curr_idx], diff
        )
        _alpha = jnp.exp(exponent) * opacity
        return _alpha, color

    alphas, colors = jax.vmap(get_alpha_and_color)(jnp.arange(max_intersects))
    trans = jnp.nancumprod(jnp.roll(1 - alphas, 1, axis=0).at[0].set(1.0), axis=0)
    img = jnp.sum(
        colors[..., None, None, :]
        * alphas[..., None]
        * trans[..., None]
        * (intersection >= 0)[..., None, None, None],
        axis=0,
    )
    img = img.clip(0, 1)
    return img
