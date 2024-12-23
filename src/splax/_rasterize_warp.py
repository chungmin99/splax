# Attempt at implementing rasterization using Warp's JAX support.
# Recall that warp, triton, etc is for writing faster CUDA code,
# and does not provide autodiff by default.

# TODO verify that the fwd/bwd is correct for small gaussians.

import jax.numpy as jnp

# Note that default `wp.launch` doesn't work with JAX jitting.
# See https://github.com/NVIDIA/warp/issues/91 for more details.
import warp as wp
from warp.jax_experimental import jax_kernel

from ._gaussian_splat import Gaussian2D
from ._camera import Camera
from jax import custom_vjp


def _rasterize_tile_warp(
    g2d: Gaussian2D,
    camera: Camera,
    depth: jnp.ndarray,
    tile_size: int,
) -> jnp.ndarray:
    assert g2d.verfify_shape() == 2
    assert len(g2d.get_batch_axes()) == 1

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


@custom_vjp
def _jax_rasterize_tile_kernel(
    indices: jnp.ndarray,
    means: jnp.ndarray,
    colors: jnp.ndarray,
    opacities: jnp.ndarray,
    inv_covs: jnp.ndarray,
    depth: jnp.ndarray,
    bbox: jnp.ndarray,
) -> jnp.ndarray:
    return jax_kernel(_rasterize_tile_kernel_fwd)(
        indices, means, colors, opacities, inv_covs, depth, bbox
    )


def _jax_rasterize_tile_kernel_fwd(
    indices: jnp.ndarray,
    means: jnp.ndarray,
    colors: jnp.ndarray,
    opacities: jnp.ndarray,
    inv_covs: jnp.ndarray,
    depth: jnp.ndarray,
    bbox: jnp.ndarray,
) -> tuple[jnp.ndarray, tuple]:
    img = _jax_rasterize_tile_kernel(
        indices, means, colors, opacities, inv_covs, depth, bbox
    )
    return img, (indices, means, colors, opacities, inv_covs, depth, bbox)


def _jax_rasterize_tile_kernel_bwd(
    res: tuple,
    grads: jnp.ndarray,
) -> tuple:
    raise NotImplementedError("Backward pass not implemented.")


_jax_rasterize_tile_kernel.defvjp(
    _jax_rasterize_tile_kernel_fwd, _jax_rasterize_tile_kernel_bwd
)


@wp.kernel
def _rasterize_tile_kernel_fwd(
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
        # Bounding box culling math.
        if bbox[idx][2] < indices[i, j, k][0] or bbox[idx][0] > indices[i, j, k][0]:
            continue
        if bbox[idx][3] < indices[i, j, k][1] or bbox[idx][1] > indices[i, j, k][1]:
            continue

        # Depth culling math.
        if depth[idx] <= 0.0:
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
