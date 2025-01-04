# Doesn't really work yet, but it's a start...

import time
from typing import cast, Literal
from pathlib import Path
import tqdm
import matplotlib.pyplot as plt
import jax_dataclasses as jdc

import jax

jax.config.update("jax_debug_nans", True)
jax.config.update("jax_debug_infs", True)
import jaxlie
import jax.numpy as jnp
import jax.scipy as jsp

import numpy as onp
import optax

from splax import Gaussian3D, Camera, rasterize

from splax._extras import Parser, Dataset

import viser
import viser.transforms as tf


def main(
    scene: Literal["garden"] = "garden",
    n_gauss: int = int(1e5),
):
    scene_dir = Path(__file__).parent / "assets" / "colmap_garden"
    parser = Parser(scene_dir)
    dataset = Dataset(parser)

    n_gauss = len(parser.points) # * 4
    gaussians = Gaussian3D.from_random(n_gauss, jax.random.PRNGKey(0))
    with jdc.copy_and_mutate(gaussians, validate=False) as gaussians:
        # gaussians.means = gaussians.means * 2.0
        gaussians.means = jnp.array(parser.points) # .repeat(4, axis=0)
        gaussians.colors = jnp.array(parser.points_rgb / 255.0) # .repeat(4, axis=0)

    print("training on", len(gaussians.means), "gaussians")

    server = viser.ViserServer()
    rendering = False

    @server.on_client_connect
    def _(cb: viser.ClientHandle):
        def update(camera: viser.CameraHandle):
            nonlocal rendering
            if rendering:
                return
            rendering = True
            width = 1080
            height = int(width / camera.aspect)
            fx = (width / 2.0) / jnp.tan(camera.fov / 2.0)
            fy = (height / 2.0) / jnp.tan(camera.fov / 2.0)
            cx = width / 2.0
            cy = height / 2.0
            K = jnp.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            viewmat = jaxlie.SE3.from_rotation_and_translation(
                jaxlie.SO3(jnp.array(camera.wxyz)),
                camera.position,
            ).as_matrix()
            _camera = Camera.from_K_and_viewmat(K, viewmat, width, height)
            g2d, depth = _camera.project(gaussians)
            img = rasterize(g2d, depth, img_width=width, img_height=height, mode="warp")
            server.scene.set_background_image(onp.array(img))
            rendering = False

        cb.camera.on_update(update)
        while True:
            update(cb.camera)
            time.sleep(0.1)

    # breakpoint()

    gui_reset_up = server.gui.add_button(
        "Reset up direction",
        hint="Set the camera control 'up' direction to the current camera's 'up'.",
    )

    @gui_reset_up.on_click
    def _(event: viser.GuiEvent) -> None:
        client = event.client
        assert client is not None
        client.camera.up_direction = tf.SO3(client.camera.wxyz) @ onp.array(
            [0.0, -1.0, 0.0]
        )

    # Define a loss function.
    import functools

    ssim_fn = jax.jit(functools.partial(compute_ssim, max_val=1.0))

    @jax.jit
    def loss_fn(gs, camera, target_img):
        g2d, depth = camera.project(gs)
        img = rasterize(g2d, depth, img_width=width, img_height=height)
        loss = jnp.abs(img - target_img).mean()
        loss = loss + (1 - ssim_fn(img, target_img))
        return loss

    @jax.jit
    def step_fn(gs, camera, target_img, opt_state):
        loss, grads = jax.value_and_grad(loss_fn)(gs, camera, target_img)
        updates, opt_state = optimizer.update(grads, opt_state)
        gs = cast(Gaussian3D, optax.apply_updates(gs, updates))
        gs = gs.fix()
        return gs, opt_state, loss

    # Initialize optimizer.
    optimizer = optax.chain(optax.adam(learning_rate=1e-2))
    opt_state = optimizer.init(gaussians)
    # optimizer = optax.adam(learning_rate=1e-2)
    # optimizer = optax.multi_transform(
    #     {
    #         "means": optax.clip(1e-1),
    #         "scale": optax.clip(1e-1),
    #         "colors": optax.clip(1e-1),
    #         "opacity": optax.clip(1e-1),
    #     },
    #     # lambda x: x,
    #     ("means", "scale", "colors", "opacity"),
    # )
    # breakpoint()
    # opt_state = optimizer.init(jax.tree.leaves(gaussians))

    # Training loop
    n_steps = 5000
    indices = jax.random.permutation(jax.random.PRNGKey(0), len(dataset))
    height, width, _ = dataset[0]["image"].shape
    pbar = tqdm.trange(n_steps)
    for step in pbar:
        curr = dataset[indices[step % len(dataset)]]
        cam = Camera.from_K_and_viewmat(curr["K"], curr["camtoworld"], width, height)
        target_img = curr["image"] / 255.0
        gaussians, opt_state, loss = step_fn(gaussians, cam, target_img, opt_state)

        if any(jnp.isnan(_).any() for _ in jax.tree.leaves(gaussians)):
            breakpoint()

        pbar.set_postfix(loss=f"{loss.item():.6f}")
        if step % 10 == 0:
            g2d, depth = cam.project(gaussians)
            img = rasterize(g2d, depth, img_width=width, img_height=height, mode="warp")
            plt.imsave("foo.png", onp.array(img))

    breakpoint()


def compute_ssim(
    img0,
    img1,
    max_val,
    filter_size=11,
    filter_sigma=1.5,
    k1=0.01,
    k2=0.03,
    return_map=False,
):
    # Taken from the mipnerf github codebase.
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


if __name__ == "__main__":
    main()
