# Attempt at implementing rasterization using Warp's JAX support.
# Recall that warp, triton, etc is for writing faster CUDA code,
# and does not provide autodiff by default.

# TODO verify that the fwd/bwd is correct for small gaussians.

import jax_dataclasses as jdc
import jax.numpy as jnp

# Note that default `wp.launch` doesn't work with JAX jitting.
# See https://github.com/NVIDIA/warp/issues/91 for more details.
import warp as wp
from warp.jax_experimental import jax_kernel

from ._gaussian_splat import Gaussian2D
from jax import custom_vjp


def _rasterize_tile_warp(
    g2d: Gaussian2D,
    tile: jnp.ndarray,
    tile_size: int,
    depth: jnp.ndarray,
) -> jnp.ndarray:
    assert len(g2d.get_batch_axes()) == 1

    inv_covs = jnp.einsum(
        "...ij,...j,...kj->...ik",
        g2d.quat.as_matrix(),
        1 / (g2d.scale**2),
        g2d.quat.as_matrix(),
    )
    bbox = g2d.get_bbox()

    indices = (
        jnp.stack(
            jnp.meshgrid(
                jnp.arange(tile_size, dtype=jnp.float32) + tile[0],
                jnp.arange(tile_size, dtype=jnp.float32) + tile[1],
            ),
            axis=-1,
        )
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


@jdc.jit
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


@jdc.jit
def _jax_rasterize_tile_kernel_bwd(
    res: tuple,
    grads: tuple,
) -> tuple:
    indices, means, colors, opacities, inv_covs, depth, bbox = res
    img_grad, alpha_grad, trans_grad = grads
    # return (indices, means, colors, opacities, inv_covs, depth, bbox)
    return (
        jnp.zeros_like(indices),
        jnp.zeros_like(means),
        jnp.zeros_like(colors),
        jnp.zeros_like(opacities),
        jnp.zeros_like(inv_covs),
        jnp.zeros_like(depth),
        jnp.zeros_like(bbox),
    )
    raise NotImplementedError("Backward pass not implemented.")


_jax_rasterize_tile_kernel.defvjp(
    _jax_rasterize_tile_kernel_fwd, _jax_rasterize_tile_kernel_bwd
)


@wp.kernel
def _rasterize_tile_kernel_fwd(
    indices: wp.array2d(dtype=wp.vec2),  # (tile_size, tile_size, 2)
    means: wp.array(dtype=wp.vec2),
    colors: wp.array(dtype=wp.vec3),
    opacities: wp.array(dtype=wp.float32),
    inv_covs: wp.array(dtype=wp.mat22f),
    depth: wp.array(dtype=wp.float32),
    bbox: wp.array(dtype=wp.vec4),
    *,
    img: wp.array2d(dtype=wp.vec3),
    alpha: wp.array2d(dtype=wp.float32),
    trans: wp.array2d(dtype=wp.float32),
):
    # Rasterize gaussians per-pixel.
    i, j = wp.tid()

    img[i, j] = wp.vec3(0.0, 0.0, 0.0)
    alpha[i, j] = 0.0
    trans[i, j] = 1.0

    for idx in range(means.shape[0]):
        # Bounding box culling math.
        if bbox[idx][2] < indices[i, j][0] or bbox[idx][0] > indices[i, j][0]:
            continue
        if bbox[idx][3] < indices[i, j][1] or bbox[idx][1] > indices[i, j][1]:
            continue

        # Depth culling math.
        if depth[idx] <= 0.0:
            continue

        mean = means[idx]
        color = colors[idx]
        opacity = opacities[idx]
        inv_cov = inv_covs[idx]

        diff = indices[i, j] - mean
        exponent = -0.5 * wp.dot(diff, wp.mul(inv_cov, diff))
        _alpha = wp.exp(exponent) * opacity

        img[i, j] += color * _alpha * trans[i, j]
        alpha[i, j] += _alpha
        trans[i, j] *= 1.0 - _alpha

        if trans[i, j] < 1e-6:
            break

    img[i, j][0] = wp.clamp(img[i, j][0], 0.0, 1.0)
    img[i, j][1] = wp.clamp(img[i, j][1], 0.0, 1.0)
    img[i, j][2] = wp.clamp(img[i, j][2], 0.0, 1.0)
    alpha[i, j] = wp.clamp(alpha[i, j], 0.0, 1.0)
