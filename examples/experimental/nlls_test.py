import jax_dataclasses as jdc
import time
import tqdm
import matplotlib.pyplot as plt

import jax
import jaxlie
import jax.numpy as jnp

import jaxls

from splax import Gaussian2D, Camera, rasterize, Gaussian3D
from splax._rasterizer import _rasterize_tile_jax_fn, _get_intersections, _rasterize_tile_jax

def train(
    camera: Camera,
    target_img: jnp.ndarray,
    tile_size: jdc.Static[int] = 10,
    max_intersects: jdc.Static[int] = 10,
    n_gaussians: jdc.Static[int] = int(1e2),
):
    tiles = jnp.stack(
        jnp.meshgrid(
            jnp.arange(camera.width // tile_size),
            jnp.arange(camera.height // tile_size),
        ),
        axis=-1,
    ).reshape(-1, 2)
    tiles = jnp.concatenate([tiles * tile_size, (tiles + 1) * tile_size], axis=-1)

    class GaussianVar(
        jaxls.Var[Gaussian2D],
        default_factory=lambda: Gaussian2D.from_random(1, jax.random.PRNGKey(0)),
        tangent_dim=Gaussian2D.tangent_dim,
        retract_fn=Gaussian2D.retract_fn,
    ): ...

    def img_loss(
        vals: jaxls.VarValues,
        var: GaussianVar,
        tile_idx: jax.Array
    ):
        gaussians = vals[var]
        tile = tiles[tile_idx][0]
        intersection = intersections[tile_idx][0]
        render = _rasterize_tile_jax_fn(gaussians, tile, tile_size, intersection)
        target_img_tile = target_img[tile_idx][0]
        loss = jnp.abs(render - target_img_tile)
        return loss.flatten()

    g2d = Gaussian2D.from_random(n_gaussians, jax.random.PRNGKey(0))
    depth = jnp.ones((n_gaussians,))
    intersections = _get_intersections(g2d, depth, tiles, max_intersects)

    g2d = jax.tree.map(lambda x: x[None, ...], g2d)

    vals = GaussianVar(jnp.arange(n_gaussians))
    factors = []
    for i in range(10):
        factors.append(
            jaxls.Factor(
                img_loss,
                (GaussianVar(i), jnp.array([i])),
            )
        )

    breakpoint()
    graph = jaxls.FactorGraph.make(
        factors,
        [vals],
        use_onp=False,
    )
    solution = graph.solve(
        initial_vals=jaxls.VarValues.make([vals.with_value(g2d)]),
        trust_region=jaxls.TrustRegionConfig(),
        termination=jaxls.TerminationConfig(
            gradient_tolerance=1e-5,
            parameter_tolerance=1e-5,
        ),
        verbose=False,
    )
    return solution[GaussianVar(jnp.arange(n_gaussians))]


def main():
    width = height = 100
    fx = fy = jnp.array(width / 4)
    cx = cy = jnp.array(width / 2)
    near = jnp.array(0.1)
    far = jnp.array(1000.0)
    pose = jaxlie.SE3.from_translation(jnp.array([0, 0, -2.0]))
    camera = Camera.from_intrinsics(fx, fy, cx, cy, width, height, near, far, pose)

    # Get target image.
    means = jnp.array([[0, 0, 0]])
    scale = jnp.array([[1, 1, 1]])
    colors = jnp.array([[0.5, 0, 0]])
    opacity = jnp.array([0.5])
    quat = jaxlie.SO3.identity((1,))
    gaussians = Gaussian3D.from_props(means, quat, scale, colors, opacity)
    g2d, depth = camera.project(gaussians)
    # target_img = rasterize(camera, g2d, depth)
    tile_size = 10
    target_img = _rasterize_tile_jax(g2d, camera, depth, tile_size, 100)

    sol = train(camera, target_img)
    breakpoint()

if __name__ == "__main__":
    main()