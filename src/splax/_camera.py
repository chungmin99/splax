from __future__ import annotations
from typing import Optional

import jax_dataclasses as jdc
import jaxlie
import jax.numpy as jnp

from ._gaussian_splat import Gaussian2D, Gaussian3D


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

    @staticmethod
    def from_K_and_viewmat(
        K: jnp.ndarray,
        T_cam_world: jnp.ndarray,
        width: jdc.Static[int],
        height: jdc.Static[int],
        near: Optional[jnp.ndarray] = None,
        far: Optional[jnp.ndarray] = None,
    ) -> Camera:
        batch_axes = K.shape[:-2]
        assert K.shape[:-2] == T_cam_world.shape[:-2]
        assert K.shape[-2:] == (3, 3)
        assert T_cam_world.shape[-2:] == (4, 4)

        if near is None:
            near = jnp.full(batch_axes, 0.1)
        if far is None:
            far = jnp.full(batch_axes, 1000.0)

        fx = K[..., 0, 0]
        fy = K[..., 1, 1]
        cx = K[..., 0, 2]
        cy = K[..., 1, 2]
        pose = jaxlie.SE3.from_matrix(T_cam_world)

        return Camera(fx, fy, cx, cy, near, far, width, height, pose)

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
        batch_axes = t.shape[:-1]
        t = t.at[..., 2].add(1e-6)  # For numerical stability.
        J = jnp.zeros((*batch_axes, 2, 3))
        J = J.at[..., 0, 0].set(self.fx / t[..., 2])
        J = J.at[..., 1, 1].set(self.fy / t[..., 2])
        J = J.at[..., 0, 2].set(-self.fx * t[..., 0] / t[..., 2] ** 2)
        J = J.at[..., 1, 2].set(-self.fy * t[..., 1] / t[..., 2] ** 2)
        return J

    @jdc.jit
    def project(self, gaussians: Gaussian3D) -> tuple[Gaussian2D, jnp.ndarray]:
        """
        Project 3D Gaussians to 2D, and also return depth from camera.
        """
        gaussians.verify_shape()

        t = self.pose.inverse() @ gaussians.means

        # If depth ~ 0, then projection to `t_d` becomes numerically unstable.
        # Setting to arbitrary position is OK since we won't render if depth < 0.
        t = jnp.where(t[..., 2:] < 1e-6, jnp.ones_like(t) * -1, t)

        t_d = jnp.einsum(
            "...ij,...j->...i",
            self.projection_mat(),
            jnp.concatenate([t, jnp.ones((*t.shape[:-1], 1))], axis=-1),
        )

        mean_d = jnp.array(
            [
                (self.width * t_d[..., 0] / (t_d[..., -1] + 1e-6) + 1) / 2 + self.cx,
                (self.height * t_d[..., 1] / (t_d[..., -1] + 1e-6) + 1) / 2 + self.cy,
            ]
        ).T

        # Project 3D covariance to 2D.
        J = self.affine_transform_jacobian(t)
        R = gaussians.quat.as_matrix()
        cov = jnp.einsum(
            "...ij,...j,...jm->...im",
            R,
            jnp.diag(gaussians.scale**2),
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

        # For numerical stability.
        cov_d = cov_d.clip(-1e6, 1e6)

        # Store covariance as scale and quaternion.
        scale, quat_d = jnp.linalg.eigh(cov_d)

        # Clip scale to prevent spurious negative values.
        # Also, this is 2D scale, so <1 means less than 1 pixel!
        scale = jnp.clip(scale, min=0.1)

        scale = jnp.sqrt(scale + 1e-6)  # For numerical stability.

        quat_d = jaxlie.SO2.from_matrix(quat_d)
        g2d = Gaussian2D.from_props(
            mean_d, quat_d, scale, gaussians.colors, gaussians.opacity
        )

        depth = t[..., 2]
        return g2d, depth
