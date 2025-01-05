from typing import Callable, Type, TypeVar, TYPE_CHECKING

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax_dataclasses as jdc
import jaxlie

if TYPE_CHECKING:
    from ._gaussian_splat import _Gaussians

T = TypeVar("T", bound="_Gaussians")


def register_gs(
    *,
    n_dim: int,
    rot_type: jdc.Static[type[jaxlie.SOBase]],
) -> Callable[[Type[T]], Type[T]]:
    """Decorator for registering Gaussian dataclasses. Copied and slightly modified from `jaxlie`.

    Sets dimensionality class variables, and marks all methods for JIT compilation.
    """

    def _wrap(cls: Type[T]) -> Type[T]:
        # Register dimensions as class attributes.
        cls.n_dim = n_dim
        cls.rot_type = rot_type

        cls.tangent_dim = (
            cls.rot_type.tangent_dim
            + cls.n_dim  # means
            + cls.n_dim  # scale
            + 3  # colors
            + 1  # opacity
        )

        # JIT all methods.
        for f in filter(
            lambda f: not f.startswith("_")
            and callable(getattr(cls, f))
            and f != "rot_type",
            dir(cls),
        ):
            setattr(cls, f, jdc.jit(getattr(cls, f)))

        return cls

    return _wrap


# Taken from the mipnerf github codebase.
def compute_ssim(
    img0,
    img1,
    max_val=1.0,
    filter_size=11,
    filter_sigma=1.5,
    k1=0.01,
    k2=0.03,
    return_map=False,
):
    """Computes SSIM from two images.

    This function was modeled after tf.image.ssim, and should produce comparable
    output.

    Args:
    img0: array. An image of size [..., width, height, num_channels].
    img1: array. An image of size [..., width, height, num_channels].
    max_val: float > 0. The maximum magnitude that `img0` or `img1` can have.
    filter_size: int >= 1. Window size.
    filter_sigma: float > 0. The bandwidth of the Gaussian used for filtering.
    k1: float > 0. One of the SSIM dampening parameters.
    k2: float > 0. One of the SSIM dampening parameters.
    return_map: Bool. If True, will cause the per-pixel SSIM "map" to returned

    Returns:
    Each image's mean SSIM, or a tensor of individual values if `return_map`.
    """
    # Construct a 1D Gaussian blur filter.
    hw = filter_size // 2
    shift = (2 * hw - filter_size + 1) / 2
    f_i = ((jnp.arange(filter_size) - hw + shift) / filter_sigma) ** 2
    filt = jnp.exp(-0.5 * f_i)
    filt /= jnp.sum(filt)

    # Blur in x and y (faster than the 2D convolution).
    def convolve2d(z, f):
        return jsp.signal.convolve2d(
            z, f, mode="valid", precision=jax.lax.Precision.HIGHEST
        )

    filt_fn1 = lambda z: convolve2d(z, filt[:, None])
    filt_fn2 = lambda z: convolve2d(z, filt[None, :])

    # Vmap the blurs to the tensor size, and then compose them.
    num_dims = len(img0.shape)
    map_axes = tuple(list(range(num_dims - 3)) + [num_dims - 1])
    for d in map_axes:
        filt_fn1 = jax.vmap(filt_fn1, in_axes=d, out_axes=d)
        filt_fn2 = jax.vmap(filt_fn2, in_axes=d, out_axes=d)
    filt_fn = lambda z: filt_fn1(filt_fn2(z))

    mu0 = filt_fn(img0)
    mu1 = filt_fn(img1)
    mu00 = mu0 * mu0
    mu11 = mu1 * mu1
    mu01 = mu0 * mu1
    sigma00 = filt_fn(img0**2) - mu00
    sigma11 = filt_fn(img1**2) - mu11
    sigma01 = filt_fn(img0 * img1) - mu01

    # Clip the variances and covariances to valid values.
    # Variance must be non-negative:
    sigma00 = jnp.maximum(0.0, sigma00)
    sigma11 = jnp.maximum(0.0, sigma11)
    sigma01 = jnp.sign(sigma01) * jnp.minimum(
        jnp.sqrt(sigma00 * sigma11 + 1e-6), jnp.abs(sigma01)
    )

    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2
    numer = (2 * mu01 + c1) * (2 * sigma01 + c2)
    denom = (mu00 + mu11 + c1) * (sigma00 + sigma11 + c2)
    ssim_map = numer / denom
    ssim = jnp.mean(ssim_map, list(range(num_dims - 3, num_dims)))
    return ssim_map if return_map else ssim
