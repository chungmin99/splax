import time
from typing import cast
import tqdm
import matplotlib.pyplot as plt

import jax
import jaxlie
import jax.numpy as jnp
import optax

from splax import Gaussian3D, Camera, rasterize

def main():
    # width = height = 1000
    n_gauss = int(1e4)

    from pathlib import Path
    # target_img = plt.imread(Path(__file__).parent / "assets/miffy.jpeg")
    # target_img = jax.image.resize(target_img, (1000, 1000, 3), method='bilinear')
    target_img = plt.imread(Path(__file__).parent / "assets/sunset.jpeg")
    target_img = jax.image.resize(target_img, (1080, 1920, 3), method='bilinear')
    height, width, _ = target_img.shape
    target_img = target_img / 255.0

    # target_img = plt.imread(Path(__file__).parent / "assets/sunset.jpeg")
    # target_img = jax.image.resize(target_img, (1080, 1920, 3), method='bilinear')

    fx = fy = jnp.array(width / 4)
    cx = cy = jnp.array(width / 2)
    near = jnp.array(0.1)
    far = jnp.array(1000.0)
    pose = jaxlie.SE3.from_translation(jnp.array([0, 0, -1.5]))
    camera = Camera.from_intrinsics(fx, fy, cx, cy, width, height, near, far, pose)

    # means = jnp.array([[0, 0, 0]])
    # scale = jnp.array([[1, 1, 1]])
    # colors = jnp.array([[0.5, 0, 0]])
    # opacity = jnp.array([0.5])
    # quat = jaxlie.SO3.identity((1,))
    # gaussians = Gaussian3D.from_props(means, quat, scale, colors, opacity)

    # g2d, depth = camera.project(gaussians)
    # target_img = rasterize(g2d, depth, img_width=width, img_height=height)
    plt.imsave("target.png", target_img)

    gaussians = Gaussian3D.from_random(n_gauss, jax.random.PRNGKey(1))

    g2d, depth = camera.project(gaussians)
    img = rasterize(g2d, depth, img_width=width, img_height=height)
    jax.block_until_ready(img)
    plt.imsave("foo.png", img)

    start_time = time.time()
    img = rasterize(g2d, depth, img_width=width, img_height=height)
    jax.block_until_ready(img)
    end_time = time.time()
    print(f"Rasterization took {end_time - start_time:.6f} seconds")

    # Define a loss function.
    @jax.jit
    def loss_fn(gs, target_img):
        g2d, depth = camera.project(gs)
        img = rasterize(g2d, depth, img_width=width, img_height=height)
        loss = jnp.abs(img - target_img).mean()
        return loss

    @jax.jit
    def step_fn(gs, opt_state):
        loss, grads = jax.value_and_grad(loss_fn)(gs, target_img)
        updates, opt_state = optimizer.update(grads, opt_state)
        gs = cast(Gaussian3D, optax.apply_updates(gs, updates))
        gs = gs.fix()
        return gs, opt_state, loss

    # Initialize optimizer.
    optimizer = optax.chain(optax.clip(1e-1), optax.adam(learning_rate=1e-2))
    opt_state = optimizer.init(gaussians)

    # Training loop
    n_steps = 1000
    pbar = tqdm.trange(n_steps)
    for step in pbar:
        gaussians, opt_state, loss = step_fn(gaussians, opt_state)
        pbar.set_postfix(loss=f"{loss.item():.6f}")
        if step % 10 == 0:
            g2d, depth = camera.project(gaussians)
            img = rasterize(g2d, depth, img_width=width, img_height=height)
            plt.imsave("foo.png", img)

if __name__ == "__main__":
    main()
