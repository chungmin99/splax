# Follow https://arxiv.org/pdf/2312.02121.
# Projection is quite fast!! It's the rasterization that needs to be optimized.
# All the activation functions!!!

from __future__ import annotations
from typing import cast
import tqdm
from typing import Optional
import matplotlib.pyplot as plt

import jax
import jax_dataclasses as jdc
import jax.numpy as jnp
import jaxlie
import optax


@jdc.pytree_dataclass
class Gaussians:
    means: jnp.ndarray
    scale: jnp.ndarray
    quat: jaxlie.SO3
    colors: jnp.ndarray
    opacity: jnp.ndarray

    @staticmethod
    def from_random(n_gaussian: int, prng_key: jax.Array) -> Gaussians:
        keys = jax.random.split(prng_key, 5)
        means = jax.random.uniform(keys[0], (n_gaussian, 3), minval=-1.0, maxval=1.0)
        scale = jnp.log(jax.random.uniform(keys[1], (n_gaussian, 3, 3), minval=0.01, maxval=0.1))
        quat = jaxlie.SO3.sample_uniform(keys[2], (n_gaussian,))
        colors = jax.random.uniform(keys[3], (n_gaussian, 3), minval=0, maxval=1)
        opacity = jax.random.uniform(keys[4], (n_gaussian,), minval=0.5, maxval=1)
        return Gaussians(means, scale, quat, colors, opacity)


@jdc.pytree_dataclass
class Gaussian2D:
    means: jnp.ndarray
    covs: jnp.ndarray
    colors: jnp.ndarray
    opacity: jnp.ndarray

    def get_bbox(self) -> jnp.ndarray:
        eigenvalues, eigenvectors = jnp.linalg.eigh(self.covs)
        radii = jnp.sqrt(eigenvalues) * 3
        extents = jnp.max(jnp.abs(radii[..., None, :] * eigenvectors), axis=-1)
        return jnp.array(
            [
                self.means[..., 0] - extents[..., 0],
                self.means[..., 1] - extents[..., 1],
                self.means[..., 0] + extents[..., 0],
                self.means[..., 1] + extents[..., 1],
            ]
        ).T

    def sort_by_depth(self, depth) -> Gaussian2D:
        indices = jnp.argsort(depth)
        return Gaussian2D(
            self.means[indices],
            self.covs[indices],
            self.colors[indices],
            self.opacity[indices],
        )


@jdc.pytree_dataclass
class Camera:
    fx: jnp.ndarray
    fy: jnp.ndarray
    cx: jnp.ndarray
    cy: jnp.ndarray
    near: jnp.ndarray
    far: jnp.ndarray
    width: jdc.Static[int]
    height: jdc.Static[int]
    pose: jaxlie.SE3
    tile_size: jdc.Static[int] = 50

    @staticmethod
    def from_intrinsics(
        fx: jnp.ndarray,
        fy: jnp.ndarray,
        cx: jnp.ndarray,
        cy: jnp.ndarray,
        width: int,
        height: int,
        near: Optional[jnp.ndarray] = None,
        far: Optional[jnp.ndarray] = None,
        pose: Optional[jaxlie.SE3] = None,
    ) -> Camera:
        batch_axes = fx.shape
        assert fx.shape == fy.shape == cx.shape == cy.shape == batch_axes
        if near is None:
            near = jnp.full(batch_axes, 0.1)
        if far is None:
            far = jnp.full(batch_axes, 1000.0)
        if pose is None:
            pose = jaxlie.SE3.identity(batch_axes=batch_axes)
        return Camera(fx, fy, cx, cy, near, far, width, height, pose)

    def projection_mat(self) -> jnp.ndarray:
        batch_axes = self.fx.shape
        P = jnp.zeros((*batch_axes, 4, 4))
        P = P.at[..., 0, 0].set(2 * self.fx / self.width)
        P = P.at[..., 1, 1].set(2 * self.fy / self.height)
        P = P.at[..., 2, 2].set((self.far + self.near) / (self.far - self.near))
        P = P.at[..., 2, 3].set(-2 * self.far * self.near / (self.far - self.near))
        P = P.at[..., 3, 2].set(1)
        return P

    def affine_transform_jacobian(self, t: jnp.ndarray) -> jnp.ndarray:
        # batch_axes = self.fx.shape
        batch_axes = t.shape[:-1]
        J = jnp.zeros((*batch_axes, 2, 3))
        J = J.at[..., 0, 0].set(self.fx / t[..., 2])
        J = J.at[..., 1, 1].set(self.fy / t[..., 2])
        J = J.at[..., 0, 2].set(-self.fx * t[..., 0] / t[..., 2] ** 2)
        J = J.at[..., 1, 2].set(-self.fy * t[..., 1] / t[..., 2] ** 2)
        return J

    @jdc.jit
    def project(self, gs: Gaussians) -> Gaussian2D:
        # Put the 3D points to camera frame
        t = self.pose.inverse() @ gs.means
        t_d = jnp.einsum(
            "...ij,...j->...i",
            self.projection_mat(),
            jnp.concatenate([t, jnp.ones((*t.shape[:-1], 1))], axis=-1),
        )

        # Project the mean.
        mean_d = jnp.array(
            [
                (self.width * t_d[..., 0] / t_d[..., -1] + 1) / 2 + self.cx,
                (self.height * t_d[..., 1] / t_d[..., -1] + 1) / 2 + self.cy,
            ]
        ).T

        # Project the covariance.
        J = self.affine_transform_jacobian(t)
        R = gs.quat.as_matrix()
        scale = jnp.exp(gs.scale)
        opacity = jax.nn.sigmoid(gs.opacity)
        # scale = gs.scale
        cov = jnp.einsum(
            "...ij,...jk,...kl,...lm->...im",
            R,
            scale,
            scale.swapaxes(-2, -1),
            R.swapaxes(-1, -2),
        )
        R_cw = self.pose.inverse().rotation().as_matrix()
        cov_d = jnp.einsum(
            "...ij,...jk,...kl,...lm,...mn->...in",
            J,
            R_cw,
            cov,
            R_cw.swapaxes(-1, -2),
            J.swapaxes(-1, -2),
        )

        g2d = Gaussian2D(mean_d, cov_d, gs.colors, opacity)

        depth = t[..., 2]
        g2d = g2d.sort_by_depth(depth)

        return g2d

    @jdc.jit
    def rasterize(
        self, g2d: Gaussian2D, max_intersects: jdc.Static[int] = 100
    ) -> jnp.ndarray:
        tiles = jnp.stack(
            jnp.meshgrid(
                jnp.arange(self.width // self.tile_size),
                jnp.arange(self.height // self.tile_size),
            ),
            axis=-1,
        ).reshape(-1, 2)
        tiles = jnp.concatenate(
            [tiles * self.tile_size, (tiles + 1) * self.tile_size], axis=-1
        )
        img_tiles = jax.vmap(
            lambda tile: self.rasterize_tile(g2d, tile, max_intersects)
        )(tiles)
        img = img_tiles.reshape(
            self.height // self.tile_size,
            self.width // self.tile_size,
            self.tile_size,
            self.tile_size,
            3,
        )
        img = img.transpose(0, 2, 1, 3, 4)
        img = img.reshape(self.height, self.width, 3)
        return img

    def rasterize_tile(self, g2d, tile, max_intersects) -> jnp.ndarray:
        bbox = g2d.get_bbox()
        in_bounds = jnp.logical_and(
            jnp.logical_and(bbox[:, 2] >= tile[0], bbox[:, 0] <= tile[2]),
            jnp.logical_and(bbox[:, 3] >= tile[1], bbox[:, 1] <= tile[3]),
        )
        indices = jnp.stack(
            jnp.meshgrid(
                jnp.arange(self.tile_size) + tile[0],
                jnp.arange(self.tile_size) + tile[1],
            ),
            axis=-1,
        )

        intersection = jnp.nonzero(in_bounds, size=max_intersects, fill_value=-1)[0]
        inv_covs = jax.vmap(jnp.linalg.inv)(g2d.covs[intersection])

        def get_alpha_and_color(idx):
            mean = g2d.means[intersection[idx]]
            color = g2d.colors[intersection[idx]]
            opacity = g2d.opacity[intersection[idx]]
            diff = indices - mean[None, None, :2]
            exponent = -0.5 * jnp.einsum(
                "...j,...jk,...k->...", diff, inv_covs[idx], diff
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


def main():
    # Test
    fx = fy = jnp.array(100)
    cx = cy = jnp.array(200)
    width = height = 400
    near = jnp.array(0.1)
    far = jnp.array(1000.0)
    pose = jaxlie.SE3.from_translation(jnp.array([0, 0, -2.0]))
    camera = Camera.from_intrinsics(fx, fy, cx, cy, width, height, near, far, pose)

    # Random gaussian initialization.
    n_gaussians = int(1e6)
    gs = Gaussians.from_random(n_gaussians, jax.random.PRNGKey(0))

    # Target image: just one red gaussian in the center.
    means = jnp.array([[0, 0, 2]])
    scale = jnp.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]])
    colors = jnp.array([[0.5, 0, 0]])
    _gs = Gaussians(means, scale, jaxlie.SO3.identity((1,)), colors, jnp.array([1.0]))
    g2d = camera.project(gs=_gs)
    target_img = camera.rasterize(g2d)
    plt.imsave("target.png", target_img)
    breakpoint()

    # Define a loss function.
    @jdc.jit
    def loss_fn(gs, target_img):
        g2d = camera.project(gs=gs)
        img = camera.rasterize(g2d)
        loss = jnp.abs(img - target_img).mean()
        return loss

    @jdc.jit
    def step_fn(gs, opt_state):
        loss, grads = jax.value_and_grad(loss_fn)(gs, target_img)
        updates, opt_state = optimizer.update(grads, opt_state)
        gs = cast(Gaussians, optax.apply_updates(gs, updates))
        return gs, opt_state, loss

    # Initialize optimizer.
    optimizer = optax.chain(optax.clip(1e-1), optax.adam(learning_rate=1e-1))
    opt_state = optimizer.init(gs)  # type: ignore

    # Training loop
    n_steps = 1000
    pbar = tqdm.trange(n_steps)
    for step in pbar:
        gs, opt_state, loss = step_fn(gs, opt_state)
        pbar.set_postfix(loss=f"{loss.item():.6f}")
        if step % 5 == 0:
            g2d = camera.project(gs)
            img = camera.rasterize(g2d)
            plt.imsave("foo.png", img)
            breakpoint()

if __name__ == "__main__":
    main()