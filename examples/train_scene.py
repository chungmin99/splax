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

from splax import Gaussian3D, Camera, rasterize, compute_ssim

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
        gaussians.scale = jnp.full(gaussians.scale.shape, 0.01)

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
            width = 720
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
            img = rasterize(g2d, depth, img_width=width, img_height=height)
            server.scene.set_background_image(onp.array(img))
            rendering = False

        cb.camera.on_update(update)

        @server.on_client_disconnect
        def _(cb: viser.ClientHandle):
            cb.camera._state.camera_cb.remove(update)

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

    @jax.jit
    def loss_fn(gs, camera, target_img):
        g2d, depth = camera.project(gs)
        img = rasterize(g2d, depth, img_width=width, img_height=height)
        loss = 0.8 * jnp.abs(img - target_img).mean()
        loss = 0.2 * (1 - compute_ssim(img, target_img))

        loss = loss + 1e-2 * jnp.abs(gaussians.opacity).mean()
        loss = loss + 1e-2 * jnp.abs(gaussians.scale).mean()
        return loss

    @jax.jit
    def step_fn(gs, camera, target_img, opt_state):
        loss, grads = jax.value_and_grad(loss_fn)(gs, camera, target_img)
        updates, opt_state = optimizer.update(grads, opt_state)
        gs = cast(Gaussian3D, optax.apply_updates(gs, updates))
        gs = gs.fix()
        return gs, opt_state, loss

    # Initialize optimizer.
    # optimizer = optax.chain(optax.adam(learning_rate=1e-2))
    optimizer = optax.multi_transform(
        {
            "means": optax.adam(learning_rate=1e-4),
            "quats": optax.adam(learning_rate=0.001),
            "scale": optax.adam(learning_rate=0.005),
            "colors": optax.adam(learning_rate=0.002),
            "opacity": optax.adam(learning_rate=0.05),
        },
        Gaussian3D(
            means="means",
            quat="quats",
            _scale="scale",
            _colors="colors",
            _opacity="opacity",
        ),
    )
    opt_state = optimizer.init(gaussians)

    # Training loop
    n_steps = 5000
    indices = jax.random.permutation(jax.random.PRNGKey(0), len(dataset))
    height, width, _ = dataset[0]["image"].shape
    prng_key = jax.random.PRNGKey(0)
    curr = dataset[0]
    cam = Camera.from_K_and_viewmat(curr["K"], curr["camtoworld"], width, height)
    g2d, depth = cam.project(gaussians)
    img = rasterize(g2d, depth, img_width=width, img_height=height)
    # from splax._rasterizer import _get_intersection
    # foo = _get_intersection(g2d, depth, jnp.array([0, 0, 100, 100]), 10, 100)
    plt.imsave('foo.png', onp.array(img[0]))
    # print(jax.vmap(lambda x: jnp.unique(x).shape[0])(img[1]))
    # for i in range(len(img[1])):
    #     print(jnp.unique(img[1][i]).shape[0])
    num_hits = [jnp.unique(img[1][i]).shape[0] for i in range(len(img[1]))]
    breakpoint()

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
            # img = rasterize(g2d, depth, img_width=width, img_height=height, mode="warp")
            img = rasterize(g2d, depth, img_width=width, img_height=height)
            plt.imsave("foo.png", onp.array(img))

        # if step % 100 == 0 and step > 0:
        #     resample_rate = 0.2
        #     gaussians, opt_state = relocate_mcmc(
        #         gaussians,
        #         opt_state,
        #         n_gauss,
        #         resample_rate,
        #         jax.random.PRNGKey(step),
        #     )

        # # Perturb gaussians.
        # gaussians = perturb_mcmc(gaussians, prng_key)

        # if step % 100 == 0:
        #     print(gaussians.opacity.min())
        # if step > 0 and step % 100 == 0:
        #     ...
    
    while True:
        clients = list(server.get_clients().values())
        if len(clients) == 0:
            time.sleep(0.1)
            continue
        camera = clients[-1].camera
        width = 720
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

    # breakpoint()

@jdc.jit
def relocate_mcmc(
    gs: Gaussian3D,
    opt_state: optax.Params,
    n_gauss: jdc.Static[int],
    resample_rate: jdc.Static[float],
    prng_key,
) -> tuple[Gaussian3D, optax.Params]:
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
    # dead_mask = jax.random.categorical(prng_key, jnp.log(1 - op), shape=(n_resample,))
    dead_mask = jnp.where(op < 0.005, size=n_resample, fill_value=-1)[0]
    dead_mask = jnp.where(
        dead_mask == -1,
        jax.random.choice(prng_key, n_gauss, shape=(n_resample,), replace=False),
        dead_mask,
    )
    moveto_mask = jax.random.choice(
        prng_key, n_gauss, shape=(n_resample,), replace=False
    )
    # moveto_mask = jax.random.categorical(prng_key, jnp.log(op**2), shape=(n_resample,))

    with jdc.copy_and_mutate(gs) as gs:
        gs.means = gs.means.at[dead_mask].set(gs.means[moveto_mask])
        gs.colors = gs.colors.at[dead_mask].set(gs.colors[moveto_mask])
        quat_params = gs.quat.parameters()
        gs.quat = gs.rot_type(quat_params.at[dead_mask].set(quat_params[moveto_mask]))

        # Update opacity and scale, accounting for gaussian composition.
        # Assume that N == 2, in equation 9 of MCMC paper (all moveto gs are unique).
        # updated_opacity = 1 - jnp.sqrt(1 - gs.opacity[moveto_mask])
        # breakpoint()
        updated_opacity = gs.opacity[moveto_mask] / 2.0
        gs.opacity = gs.opacity.at[dead_mask].set(updated_opacity)
        gs.opacity = gs.opacity.at[moveto_mask].set(updated_opacity)

        # updated_scale = jnp.pow(
        #     (
        #         gs.opacity  # 0 choose 0
        #         + gs.opacity  # 1 choose 0
        #         + -1 * gs.opacity**2 / jnp.sqrt(2.0)  # 1 choose 1
        #     ),
        #     -2.0,
        # )
        # updated_scale = jnp.einsum(
        #     "...,...,...j->...j",
        #     gs.opacity**2,
        #     updated_scale,
        #     gs.scale,
        # )[moveto_mask]
        updated_scale = gs.scale[moveto_mask]
        gs.scale = gs.scale.at[dead_mask].set(updated_scale)
        gs.scale = gs.scale.at[moveto_mask].set(updated_scale)

    # Reset optimizer state.
    opt_state = jax.tree.map(
        lambda x: x
        if len(x.shape) == 0 or x.shape[0] != n_gauss
        else x.at[moveto_mask].set(0.0),
        opt_state,
    )

    return gs, opt_state


@jdc.jit
def perturb_mcmc(gaussians, prng_key):
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
    main()
