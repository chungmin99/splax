import tqdm
import matplotlib.pyplot as plt

import jax
import jaxlie
import jax.numpy as jnp
import optax

from splax import Gaussians, Camera, Rasterizer

fx = fy = jnp.array(100*2)
cx = cy = jnp.array(200*2)
width = height = 400*2
near = jnp.array(0.1)
far = jnp.array(1000.0)
pose = jaxlie.SE3.from_translation(jnp.array([0, 0, -2.0]))
camera = Camera.from_intrinsics(fx, fy, cx, cy, width, height, near, far, pose)

means = jnp.array([[0, 0, 0]])
scale = jnp.array([[1, 1, 1]])
colors = jnp.array([[0.5, 0, 0]])
opacity = jnp.array([0.5])
quat = jaxlie.SO3.identity((1,))
gaussians = Gaussians.from_props(means, quat, scale, colors, opacity)
rast = Rasterizer(40, 100)

g2d, depth = camera.project(gaussians)
target_img = rast.rasterize(camera, g2d, depth)

n_gauss = int(1e4)
gaussians = Gaussians.from_random(n_gauss, jax.random.PRNGKey(1), 3)

g2d, depth = camera.project(gaussians)
img = rast.rasterize(camera, g2d, depth)
plt.imsave("foo.png", img)

# Define a loss function.
@jax.jit
def loss_fn(gs, target_img):
    g2d, depth = camera.project(gs)
    img = rast.rasterize(camera, g2d, depth)
    loss = jnp.abs(img - target_img).mean()
    return loss

@jax.jit
def step_fn(gs, opt_state):
    loss, grads = jax.value_and_grad(loss_fn)(gs, target_img)
    updates, opt_state = optimizer.update(grads, opt_state)
    gs = optax.apply_updates(gs, updates)
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
    if step % 5 == 0:
        g2d, depth = camera.project(gaussians)
        img = rast.rasterize(camera, g2d, depth)
        plt.imsave("foo.png", img)

# plt.imsave("foo.png", arr=img)

# breakpoint()