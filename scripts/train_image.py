from typing import cast, Literal, Union
from pathlib import Path
import matplotlib.pyplot as plt
import tqdm
import imageio
import tyro

import jax
import jax.numpy as jnp
import numpy as onp
import jax_dataclasses as jdc
import optax
import jaxlie

from splax import Gaussian2D, Gaussian3D, rasterize, Camera


@jdc.jit
def normalize_to_image_space(
    g2d: Gaussian2D, height: jdc.Static[int], width: jdc.Static[int]
) -> Gaussian2D:
    with jdc.copy_and_mutate(g2d) as _g2d:
        _g2d.means = (g2d.means + 1) * jnp.array([width, height]) / 2
        _g2d._scale = jnp.log(g2d.scale * width)
    return _g2d


def main(
    n_gauss: int = int(1e5),
    n_steps: int = 1000,
    scene: Literal["miffy", "sunset"] = "sunset",
    mode: Literal["2D", "3D"] = "2D",
):
    """
    Simple gaussian splat optimization in image space.

    Can optimize as "2D" (3D gaussians in SE2, with pre-determined rasterization ordering),
    or as "3D" (3D gaussians in SE3, with perspective projection).
    """
    if scene == "miffy":
        target_img = plt.imread(Path(__file__).parent / "assets/miffy.jpeg")
        target_img = jax.image.resize(target_img, (1000, 1000, 3), method="bilinear")

    elif scene == "sunset":
        target_img = plt.imread(Path(__file__).parent / "assets/sunset.jpeg")
        target_img = jax.image.resize(target_img, (1080, 1920, 3), method="bilinear")

    target_img = target_img / 255.0
    height, width, _ = target_img.shape

    prng_key = jax.random.PRNGKey(0)

    if mode == "2D":
        gaussians = Gaussian2D.from_random(n_gauss, prng_key)

        def rasterize_gs(gs):
            _gs = normalize_to_image_space(gs, height, width)
            img = rasterize(_gs, jnp.arange(n_gauss), height, width)
            return img

    elif mode == "3D":
        fx = fy = jnp.array(width / 4)
        cx = cy = jnp.array(width / 2)
        near = jnp.array(0.1)
        far = jnp.array(1000.0)
        pose = jaxlie.SE3.from_translation(jnp.array([0, 0, -1.5]))
        camera = Camera.from_intrinsics(fx, fy, cx, cy, width, height, near, far, pose)
        gaussians = Gaussian3D.from_random(n_gauss, prng_key)

        def rasterize_gs(gs):
            _gs_2d, depth = camera.project(gs)
            img = rasterize(_gs_2d, depth, height, width)
            return img

    else:
        raise ValueError(f"Invalid mode {mode}.")

    @jdc.jit
    def loss_fn(gaussians):
        img = rasterize_gs(gaussians)
        loss = jnp.abs(img - target_img).mean()
        return loss

    @jdc.jit
    def step_fn(gaussians, opt_state):
        loss, grads = jax.value_and_grad(loss_fn)(gaussians)
        updates, opt_state = optimizer.update(grads, opt_state)
        gaussians = optax.apply_updates(gaussians, updates)
        gaussians = gaussians.fix()  # pyright: ignore
        return gaussians, opt_state, loss

    # Initialize optimizer.
    optimizer = optax.chain(optax.clip(1e-1), optax.adam(learning_rate=1e-2))
    opt_state = optimizer.init(cast(optax.Params, gaussians))

    # Training loop.
    result_imgs = []
    pbar = tqdm.trange(n_steps)
    for step in pbar:
        gaussians, opt_state, loss = step_fn(gaussians, opt_state)
        pbar.set_postfix(loss=f"{loss.item():.6f}")
        if step % 20 == 0:
            img = rasterize_gs(gaussians)
            plt.imsave("out.png", onp.array(img))
            result_imgs.append(img)

    # Save the result images as a gif.
    result_imgs = [onp.array(img * 255, dtype=onp.uint8) for img in result_imgs]
    imageio.mimsave(f"{scene}_result_{mode}.gif", result_imgs, format="GIF", loop=0, fps=10)  # pyright: ignore


if __name__ == "__main__":
    tyro.cli(main)
