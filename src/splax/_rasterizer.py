from __future__ import annotations

import jax
import jax_dataclasses as jdc
import jax.numpy as jnp

from ._gaussian_splat import Gaussians
from ._camera import Camera

# try:
#     import triton
#     import triton.language as tl
# except ImportError:
#     triton = None

@jdc.pytree_dataclass
class Rasterizer:
    tile_size: jdc.Static[int]
    max_intersects: jdc.Static[int]

    @jdc.jit
    def rasterize(self, camera: Camera, gaussians: Gaussians, depth: jnp.ndarray) -> jnp.ndarray:
        assert gaussians.get_and_check_shape() == 2
        tiles = jnp.stack(
            jnp.meshgrid(
                jnp.arange(camera.width // self.tile_size),
                jnp.arange(camera.height // self.tile_size),
            ),
            axis=-1,
        ).reshape(-1, 2)

        tiles = jnp.concatenate(
            [tiles * self.tile_size, (tiles + 1) * self.tile_size], axis=-1
        )
        img_tiles = jax.vmap(
            lambda tile: self.rasterize_tile(
                gaussians, tile, depth
            )
        )(tiles)
        img = img_tiles.reshape(
            camera.height // self.tile_size,
            camera.width // self.tile_size,
            self.tile_size,
            self.tile_size,
            3,
        )
        img = img.transpose(0, 2, 1, 3, 4)
        img = img.reshape(camera.height, camera.width, 3)
        return img

    def rasterize_tile(self, g2d: Gaussians, tile: jnp.ndarray, depth: jnp.ndarray) -> jnp.ndarray:
        bbox = g2d.get_bbox()
        in_bounds = jnp.logical_and(
            jnp.logical_and(bbox[:, 2] >= tile[0], bbox[:, 0] <= tile[2]),
            jnp.logical_and(bbox[:, 3] >= tile[1], bbox[:, 1] <= tile[3]),
        )
        in_bounds = jnp.logical_and(in_bounds, depth > 0)

        indices = jnp.stack(
            jnp.meshgrid(
                jnp.arange(self.tile_size) + tile[0],
                jnp.arange(self.tile_size) + tile[1],
            ),
            axis=-1,
        )

        intersection = jnp.nonzero(in_bounds, size=self.max_intersects, fill_value=-1)[0]
        inv_covs = jnp.einsum(
            "...ij,...j,...kj->...ik",
            g2d.quat.as_matrix(),
            1/(g2d.scale**2),
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

        alphas, colors = jax.vmap(get_alpha_and_color)(jnp.arange(self.max_intersects))
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

    # @triton.jit
    # def rasterize_tile_kernel(
    #     gaussians_means, gaussians_colors, gaussians_opacity, gaussians_inv_covs,
    #     tile, depth, img, tile_size, max_intersects, width, height
    # ):
    #     idx = tl.program_id(0)
    #     tile_x = tile[idx, 0]
    #     tile_y = tile[idx, 1]
    #     tile_w = tile[idx, 2]
    #     tile_h = tile[idx, 3]

    #     for i in range(tile_x, tile_w):
    #         for j in range(tile_y, tile_h):
    #             if depth[i, j] > 0:
    #                 for k in range(max_intersects):
    #                     mean = gaussians_means[k]
    #                     color = gaussians_colors[k]
    #                     opacity = gaussians_opacity[k]
    #                     inv_cov = gaussians_inv_covs[k]

    #                     diff_x = i - mean[0]
    #                     diff_y = j - mean[1]
    #                     exponent = -0.5 * (diff_x * inv_cov[0, 0] * diff_x + diff_y * inv_cov[1, 1] * diff_y)
    #                     alpha = tl.exp(exponent) * opacity

    #                     img[i, j, 0] += color[0] * alpha
    #                     img[i, j, 1] += color[1] * alpha
    #                     img[i, j, 2] += color[2] * alpha

    # def rasterize_tile_triton(self, g2d: Gaussians, tile: jnp.ndarray, depth: jnp.ndarray) -> jnp.ndarray:
    #     img = jnp.zeros((self.tile_size, self.tile_size, 3), dtype=jnp.float32)
    #     gaussians_means = g2d.means
    #     gaussians_colors = g2d.colors
    #     gaussians_opacity = g2d.opacity
    #     gaussians_inv_covs = jnp.linalg.inv(g2d.quat.as_matrix() @ jnp.diag(1/(g2d.scale**2)) @ g2d.quat.as_matrix())

    #     triton.launch(
    #         rasterize_tile_kernel,
    #         grid=(tile.shape[0],),
    #         args=[
    #             gaussians_means, gaussians_colors, gaussians_opacity, gaussians_inv_covs,
    #             tile, depth, img, self.tile_size, self.max_intersects, g2d.width, g2d.height
    #         ],
    #         num_warps=8
    #     )
    #     return img