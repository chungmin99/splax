from __future__ import annotations
from typing import ClassVar, Self

import jax
import jax_dataclasses as jdc
import jaxlie

import jax.numpy as jnp

from ._utils import register_gs


@jdc.pytree_dataclass
class _Gaussians:
    means: jnp.ndarray
    quat: jaxlie.SOBase
    _scale: jnp.ndarray
    _colors: jnp.ndarray
    _opacity: jnp.ndarray

    n_dim: ClassVar[int]

    @classmethod
    def from_props(
        cls,
        means: jnp.ndarray,
        quat: jaxlie.SOBase,
        scale: jnp.ndarray,
        colors: jnp.ndarray,
        opacity: jnp.ndarray,
    ) -> Self:
        _scale = jnp.log(scale)
        _colors = jax.scipy.special.logit(colors)
        _opacity = jax.scipy.special.logit(opacity)
        gaussians = cls(means, quat, _scale, _colors, _opacity)
        gaussians.verfify_shape()
        return gaussians

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
        self.verfify_shape()
        return self.means.shape[:-1]

    def verfify_shape(self):
        n_dim = self.n_dim
        batch_axes = self.means.shape[:-1]
        assert self.means.shape == (*batch_axes, n_dim)
        assert self._scale.shape == (*batch_axes, n_dim)
        assert self._colors.shape == (*batch_axes, 3)
        assert self._opacity.shape == batch_axes
        assert self.quat.get_batch_axes() == batch_axes
        assert self.quat.space_dim == n_dim

    def get_bbox(self) -> jnp.ndarray:
        cov_mat = self.quat.as_matrix()
        extent = 3 * jnp.abs(cov_mat * self.scale[..., None, :]).max(axis=-1)
        bbox = jnp.concatenate([self.means - extent, self.means + extent], axis=-1)
        return bbox

    @classmethod
    def from_random(cls, n_gauss: jdc.Static[int], prng_key: jax.Array) -> Self:
        keys = jax.random.split(prng_key, 5)

        means = jax.random.uniform(
            keys[0], (n_gauss, cls.n_dim), minval=-1.0, maxval=1.0
        )
        scale = jax.random.uniform(
            keys[1], (n_gauss, cls.n_dim), minval=0.01, maxval=0.1
        )
        colors = jax.random.uniform(keys[2], (n_gauss, 3), minval=0, maxval=1)
        opacity = jax.random.uniform(keys[3], (n_gauss,), minval=0.5, maxval=1)

        if cls.n_dim == 3:
            SOBase = jaxlie.SO3
        elif cls.n_dim == 2:
            SOBase = jaxlie.SO2
        else:
            raise ValueError(f"Unsupported space dimension: {cls.n_dim}")
        quat = SOBase.sample_uniform(keys[4], (n_gauss,))

        _scale = jnp.log(scale)
        _colors = jax.scipy.special.logit(colors)
        _opacity = jax.scipy.special.logit(opacity)

        gaussians = cls(means, quat, _scale, _colors, _opacity)
        gaussians.verfify_shape()
        return gaussians

    @classmethod
    def get_tangent_dim(cls) -> int:
        return (
            cls.quat.tangent_dim
            + cls.n_dim  # means
            + cls.n_dim  # scale
            + 3  # colors
            + 1  # opacity
        )
    
    def fix(self) -> Self:
        with jdc.copy_and_mutate(self) as g:
            g.quat = jax.tree.map(
                lambda x: x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-6), g.quat
            )
        return g

@register_gs(n_dim=3)
@jdc.pytree_dataclass
class Gaussian3D(_Gaussians): ...

@register_gs(n_dim=2)
@jdc.pytree_dataclass
class Gaussian2D(_Gaussians): ...