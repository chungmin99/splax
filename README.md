# splax

Quick winter project to see if it's possible to write a gaussian splat renderer purely in JAX. Inspired by `gsplat` issue [here](https://github.com/nerfstudio-project/gsplat/issues/175).

More specifically, I was interested in:
- if a pure JAX implementation would be fast to be useful enough,
- using a constant `max_intersects_per_tile` for JIT.

TLDR, I couldn't get it to work. Well, expectedly, it's slow by ~10-100x. 

More specifically on (2), I found that:

1. `num_intersects` needs to be quite low (~100-200) for rendering speed to feel reasonable. This is a _very_ small subset of what should actually be rendered.
2. Failing to render all the intersecting gaussians leads to "blank patches" -- that the splats will be optimized to grow into, incorrectly.

It does lead to an interesting research question (or maybe it's quite simple):
> If we needed to choose `n_render << n_intersects` gaussians to be rendered, how would we do it?

We can't just select the biggest gaussians (need texture), or the closest gaussians (may fail to include important background).
Perhaps it would be useful for low-memory requirement settings (e.g., [second order opt](https://github.com/lukasHoel/3DGS-LM)).
Or training a network on gaussians, where multiple scenes need to fit into memory?

At least I got to splat a sunset.

<img src="sunset_result_3D.gif" height=100>
