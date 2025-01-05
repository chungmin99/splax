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

from splax import Gaussian2D, Gaussian3D, rasterize, Camera, compute_ssim


def main(
    n_gauss: int = int(1e5),
    n_steps: int = 1000,
    scene: Literal["miffy", "sunset", "sunset_1080p"] = "sunset",
    mode: Literal["2D", "3D"] = "2D",
    train_mode: Literal["default", "mcmc"] = "default",
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

    elif scene == "sunset_1080p":
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
        loss = 0.8 * jnp.abs(img - target_img).mean()
        loss = 0.2 * (1 - compute_ssim(img, target_img))

        # Add scale + opacity regularization for MCMC.
        if train_mode == "mcmc":
            loss = loss + 1e-2 * jnp.abs(gaussians.opacity).mean()
            loss = loss + 1e-2 * jnp.abs(gaussians.scale).mean()

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

        # Output + save current image.
        if step % 20 == 0:
            img = rasterize_gs(gaussians)
            plt.imsave("out.png", onp.array(img))
            result_imgs.append(img)

        if train_mode == "mcmc":
            # Relocate gaussians.
            if step % 100 == 0 and step > 0:
                resample_rate = 0.05
                gaussians, opt_state = relocate_mcmc(
                    gaussians,
                    opt_state,
                    n_gauss,
                    resample_rate,
                    prng_key,
                )

            # Perturb gaussians.
            gaussians = perturb_mcmc(gaussians, prng_key)

    # Save the result images as a gif.
    result_imgs = [onp.array(img * 255, dtype=onp.uint8) for img in result_imgs]
    imageio.mimsave(
        f"{scene}_result_{mode}.gif",
        result_imgs,  # pyright: ignore
        format="GIF",  # pyright: ignore
        loop=0,
        fps=10,
    )


@jdc.jit
def normalize_to_image_space(
    g2d: Gaussian2D, height: jdc.Static[int], width: jdc.Static[int]
) -> Gaussian2D:
    with jdc.copy_and_mutate(g2d) as _g2d:
        _g2d.means = (g2d.means + 1) * jnp.array([width, height]) / 2
        _g2d._scale = jnp.log(g2d.scale * width)
    return _g2d


@jdc.jit
def relocate_mcmc(
    gs: Union[Gaussian2D, Gaussian3D],
    opt_state: optax.Params,
    n_gauss: jdc.Static[int],
    resample_rate: jdc.Static[float],
    prng_key,
) -> tuple[Union[Gaussian2D, Gaussian3D], optax.Params]:
    n_resample = int(n_gauss * resample_rate)

    # Determine which gaussians to move; here we:
    # - Randomly choose gaussians to kill, based on their opacity, and
    # - Randomly choose gaussians to move to.
    #
    # This doesn't quite follow the MCMC paper, which states that;
    # - Gaussians are dead if opacity < 0.005,
    # - We should be sampling from the alive gaussians, and
    # - Sampling based on the opacity of the gaussians.
    # But this should also technically work, and is simpler.

    op = gs.opacity
    dead_mask = jax.random.categorical(prng_key, jnp.log(1 - op), shape=(n_resample,))
    moveto_mask = jax.random.choice(
        prng_key, n_gauss, shape=(n_resample,), replace=False
    )

    with jdc.copy_and_mutate(gs) as gs:
        gs.means = gs.means.at[dead_mask].set(gs.means[moveto_mask])
        gs.colors = gs.colors.at[dead_mask].set(gs.colors[moveto_mask])
        quat_params = gs.quat.parameters()
        gs.quat = gs.rot_type(quat_params.at[dead_mask].set(quat_params[moveto_mask]))

        # Update opacity and scale, accounting for gaussian composition.
        # Assume that N == 2, in equation 9 of MCMC paper (all moveto gs are unique).
        updated_opacity = 1 - jnp.sqrt(1 - gs.opacity[moveto_mask])
        gs.opacity = gs.opacity.at[dead_mask].set(updated_opacity)
        gs.opacity = gs.opacity.at[moveto_mask].set(updated_opacity)

        updated_scale = jnp.pow(
            (
                gs.opacity  # 0 choose 0
                + gs.opacity  # 1 choose 0
                + -1 * gs.opacity**2 / jnp.sqrt(2.0)  # 1 choose 1
            ),
            -2.0,
        )
        updated_scale = jnp.einsum(
            "...,...,...j->...j",
            gs.opacity**2,
            updated_scale,
            gs.scale,
        )[moveto_mask]
        gs.scale = gs.scale.at[dead_mask].set(updated_scale)
        gs.scale = gs.scale.at[moveto_mask].set(updated_scale)

    # Reset optimizer state.
    opt_state = jax.tree.map(
        lambda x: x
        if len(x.shape) == 0 or x.shape[0] != n_gauss
        else x.at[dead_mask].set(x[moveto_mask]).at[moveto_mask].set(0.0),
        opt_state,
    )

    return gs, opt_state


@jdc.jit
def perturb_mcmc(gaussians: Union[Gaussian2D, Gaussian3D], prng_key):
    lamb_noise = 5e5

    mean_noise = lamb_noise * jnp.einsum(
        "...,...ij,...j->...j",
        jax.nn.sigmoid(-100.0 * (gaussians.opacity - 0.005)),
        gaussians.cov,
        jax.random.normal(prng_key, gaussians.means.shape),
    )  # i.e., perturb more if large spread, low opacity.

    with jdc.copy_and_mutate(gaussians) as gaussians:
        gaussians.means = gaussians.means + mean_noise
    return gaussians


if __name__ == "__main__":
    tyro.cli(main)
