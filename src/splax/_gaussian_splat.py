from __future__ import annotations
import jax
import jax_dataclasses as jdc
import jaxlie

import jax.numpy as jnp


@jdc.pytree_dataclass
class Gaussians:
    means: jnp.ndarray
    quat: jaxlie.SOBase
    _scale: jnp.ndarray
    _colors: jnp.ndarray
    _opacity: jnp.ndarray

    @staticmethod
    def from_props(
        means: jnp.ndarray,
        quat: jaxlie.SOBase,
        scale: jnp.ndarray,
        colors: jnp.ndarray,
        opacity: jnp.ndarray,
    ) -> Gaussians:
        _scale = jnp.log(scale)
        _colors = jax.scipy.special.logit(colors)
        _opacity = jax.scipy.special.logit(opacity)
        return Gaussians(means, quat, _scale, _colors, _opacity)

    @property
    def scale(self) -> jnp.ndarray:
        return jnp.exp(self._scale)

    @property
    def colors(self) -> jnp.ndarray:
        return jax.nn.sigmoid(self._colors)

    @property
    def opacity(self) -> jnp.ndarray:
        return jax.nn.sigmoid(self._opacity)
    
    def get_batch_axes(self) -> tuple[int, ...]:
        self.get_and_check_shape()
        return self.means.shape[:-1]

    def get_and_check_shape(self) -> int:
        n_dim = self.quat.space_dim
        batch_axes = self.means.shape[:-1]
        assert self.means.shape == (*batch_axes, n_dim)
        assert self._scale.shape == (*batch_axes, n_dim)
        assert self._colors.shape == (*batch_axes, 3)
        assert self._opacity.shape == batch_axes
        assert self.quat.get_batch_axes() == batch_axes
        return n_dim

    def get_bbox(self) -> jnp.ndarray:
        cov_mat = self.quat.as_matrix()
        extent = 3 * jnp.abs(cov_mat * self.scale[..., None, :]).max(axis=-1)
        bbox = jnp.concatenate([self.means - extent, self.means + extent], axis=-1)
        return bbox

    @staticmethod
    def from_random(n_gauss: int, prng_key: jax.Array, n_dim: int) -> Gaussians:
        keys = jax.random.split(prng_key, 5)

        means = jax.random.uniform(keys[0], (n_gauss, n_dim), minval=-1.0, maxval=1.0)
        scale = jax.random.uniform(keys[1], (n_gauss, n_dim), minval=0.01, maxval=0.1)
        colors = jax.random.uniform(keys[2], (n_gauss, n_dim), minval=0, maxval=1)
        opacity = jax.random.uniform(keys[3], (n_gauss,), minval=0.5, maxval=1)

        if n_dim == 3:
            SOBase = jaxlie.SO3
        elif n_dim == 2:
            SOBase = jaxlie.SO2
        else:
            raise ValueError(f"Unsupported space dimension: {n_dim}")
        quat = SOBase.sample_uniform(keys[4], (n_gauss,))

        _scale = jnp.log(scale)
        _colors = jax.scipy.special.logit(colors)
        _opacity = jax.scipy.special.logit(opacity)

        return Gaussians(means, quat, _scale, _colors, _opacity)
