# Attempt at implementing rasterization using Warp's JAX support.
# Note that `jax_kernel` does not provide autodiff by default.

import jax.numpy as jnp

# Note that default `wp.launch` doesn't work with JAX jitting.
# See https://github.com/NVIDIA/warp/issues/91 for more details.
import warp as wp
from warp.jax_experimental import jax_kernel

from ._gaussian_splat import Gaussians
from ._camera import Camera


def _rasterize_tile_warp(
    g2d: Gaussians,
    camera: Camera,
    depth: jnp.ndarray,
    tile_size: int,
) -> jnp.ndarray:
    assert g2d.get_and_check_shape() == 2
    inv_covs = jnp.einsum(
        "...ij,...j,...kj->...ik",
        g2d.quat.as_matrix(),
        1 / (g2d.scale**2),
        g2d.quat.as_matrix(),
    )
    bbox = g2d.get_bbox()

    n_tile_h = camera.height // tile_size
    n_tile_w = camera.width // tile_size
    indices = (
        jnp.stack(
            jnp.meshgrid(
                jnp.arange(camera.height, dtype=jnp.float32),
                jnp.arange(camera.width, dtype=jnp.float32),
            ),
            axis=-1,
        )
        .reshape(n_tile_h, tile_size, n_tile_w, tile_size, 2)
        .transpose(0, 2, 1, 3, 4)
        .reshape(n_tile_h * n_tile_w, tile_size, tile_size, 2)
    )
    output = _jax_rasterize_tile_kernel(
        indices,
        g2d.means,
        g2d.colors,
        g2d.opacity,
        inv_covs,
        depth,
        bbox,
    )
    img = output[0]
    return img


@wp.kernel
def _rasterize_tile_kernel(
    indices: wp.array3d(dtype=wp.vec2),  # (n_tiles, tile_size, tile_size, 2)
    means: wp.array(dtype=wp.vec2),
    colors: wp.array(dtype=wp.vec3),
    opacities: wp.array(dtype=wp.float32),
    inv_covs: wp.array(dtype=wp.mat22f),
    depth: wp.array(dtype=wp.float32),
    bbox: wp.array(dtype=wp.vec4),
    *,
    img: wp.array3d(dtype=wp.vec3),
    alpha: wp.array3d(dtype=wp.float32),
    trans: wp.array3d(dtype=wp.float32),
):
    # Rasterize gaussians per-pixel.
    i, j, k = wp.tid()

    img[i, j, k] = wp.vec3(0.0, 0.0, 0.0)
    alpha[i, j, k] = 0.0
    trans[i, j, k] = 1.0

    for idx in range(means.shape[0]):
        if depth[idx] <= 0.0:
            continue

        # Do the culling math here!
        if bbox[idx][2] < indices[i, j, k][0] or bbox[idx][0] > indices[i, j, k][0]:
            continue
        if bbox[idx][3] < indices[i, j, k][1] or bbox[idx][1] > indices[i, j, k][1]:
            continue

        mean = means[idx]
        color = colors[idx]
        opacity = opacities[idx]
        inv_cov = inv_covs[idx]

        diff = indices[i, j, k] - mean
        exponent = -0.5 * wp.dot(diff, wp.mul(inv_cov, diff))
        _alpha = wp.exp(exponent) * opacity

        img[i, j, k] += color * _alpha * trans[i, j, k]
        alpha[i, j, k] += _alpha
        trans[i, j, k] *= 1.0 - _alpha

        if trans[i, j, k] < 1e-6:
            break

    img[i, j, k][0] = wp.clamp(img[i, j, k][0], 0.0, 1.0)
    img[i, j, k][1] = wp.clamp(img[i, j, k][1], 0.0, 1.0)
    img[i, j, k][2] = wp.clamp(img[i, j, k][2], 0.0, 1.0)
    alpha[i, j, k] = wp.clamp(alpha[i, j, k], 0.0, 1.0)


_jax_rasterize_tile_kernel = jax_kernel(_rasterize_tile_kernel)
