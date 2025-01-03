from typing import cast, Literal
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

from splax import Gaussian2D, rasterize

@jdc.jit
def normalize_to_image_space(g2d: Gaussian2D, height: jdc.Static[int], width: jdc.Static[int]) -> Gaussian2D:
    with jdc.copy_and_mutate(g2d) as _g2d:
        _g2d.means = (g2d.means + 1) * jnp.array([width, height]) / 2
        _g2d._scale = jnp.log(g2d.scale * width)
    return _g2d

def main(
    n_gauss: int = int(1e5),
    n_steps: int = 1000,
    scene: Literal["miffy", "sunset"] = "sunset",
):
    """
    Simple gaussian splat optimization in image space.
    """
    if scene == "miffy":
        target_img = plt.imread(Path(__file__).parent / "assets/miffy.jpeg")
        target_img = jax.image.resize(target_img, (1000, 1000, 3), method='bilinear')

    elif scene == "sunset":
        target_img = plt.imread(Path(__file__).parent / "assets/sunset.jpeg")
        target_img = jax.image.resize(target_img, (1080, 1920, 3), method='bilinear')

    target_img = target_img / 255.0
    height, width, _ = target_img.shape

    prng_key = jax.random.PRNGKey(0)
    g2d = Gaussian2D.from_random(n_gauss, prng_key)

    img = rasterize(g2d, jnp.arange(n_gauss), height, width)

    @jdc.jit
    def loss_fn(g2d):
        _g2d = normalize_to_image_space(g2d, height, width)
        img = rasterize(_g2d, jnp.arange(n_gauss), height, width)
        loss = jnp.abs(img - target_img).mean()
        return loss

    @jdc.jit
    def step_fn(g2d, opt_state):
        loss, grads = jax.value_and_grad(loss_fn)(g2d)
        updates, opt_state = optimizer.update(grads, opt_state)
        g2d = cast(Gaussian2D, optax.apply_updates(g2d, updates))
        g2d = g2d.fix()
        return g2d, opt_state, loss

    # Initialize optimizer.
    optimizer = optax.chain(optax.clip(1e-1), optax.adam(learning_rate=1e-2))
    opt_state = optimizer.init(cast(optax.Params, g2d))

    # Training loop.
    result_imgs = []
    pbar = tqdm.trange(n_steps)
    for step in pbar:
        g2d, opt_state, loss = step_fn(g2d, opt_state)
        pbar.set_postfix(loss=f"{loss.item():.6f}")
        if step % 20 == 0:
            _g2d = normalize_to_image_space(g2d, height, width)
            img = rasterize(_g2d, jnp.arange(n_gauss), height, width)
            result_imgs.append(img)
    
    # Save the result images as a gif.
    result_imgs = [onp.array(img * 255, dtype=onp.uint8) for img in result_imgs]
    imageio.mimsave(f"{scene}_result.gif", result_imgs, fps=10, format="GIF")  # pyright: ignore

if __name__ == "__main__":
    tyro.cli(main)