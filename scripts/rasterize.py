from typing import Literal
import matplotlib.pyplot as plt
import tyro

import jax
import jaxlie
import jax.numpy as jnp

from splax import Gaussian3D, Camera, rasterize


def main(
    mode: Literal["jax", "warp"] = "jax",
):
    width = height = 1000
    fx = fy = jnp.array(width / 4)
    cx = cy = jnp.array(width / 2)
    near = jnp.array(0.1)
    far = jnp.array(1000.0)
    pose = jaxlie.SE3.from_translation(jnp.array([0, 0, -2.0]))
    camera = Camera.from_intrinsics(fx, fy, cx, cy, width, height, near, far, pose)

    n_gauss = int(1e5)
    gaussians = Gaussian3D.from_random(n_gauss, jax.random.PRNGKey(1))

    g2d, depth = camera.project(gaussians)
    target_img = rasterize(camera, g2d, depth, mode=mode)
    plt.imsave("rast_random.png", target_img)

    means = jnp.array([[0, 0, 0]])
    scale = jnp.array([[1, 1, 1]])
    colors = jnp.array([[0.5, 0, 0]])
    opacity = jnp.array([0.5])
    quat = jaxlie.SO3.identity((1,))
    gaussians = Gaussian3D.from_props(means, quat, scale, colors, opacity)

    g2d, depth = camera.project(gaussians)
    target_img = rasterize(camera, g2d, depth, mode=mode)
    plt.imsave("rast_red_circle.png", target_img)


if __name__ == "__main__":
    tyro.cli(main)
