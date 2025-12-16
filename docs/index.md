---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  name: python3
  display_name: Python 3
---

# potamides

```{toctree}
:maxdepth: 2
:hidden:
:caption: Guides

guides/begin.md
guides/stream_fitting.md
guides/1D-inference.md
guides/2D-inference.md
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: 🔌 API Reference

api/index.md
```

# 🚀 Get Started

**potamides** is a Python package for constraining gravitational potentials
using stellar stream curvature analysis. Built on [JAX][jax], it combines
spline-based stream modeling with Bayesian inference to extract gravitational
field parameters from the geometric properties of stellar streams.

The name is inspired by Greek ποταμίδες ("potamídes", meaning "river streams"),
with the initial "P" representing $\Phi$, the conventional symbol for
gravitational potential in astronomy.

## Key Features

- 🌊 **Spline-based stream modeling**: Smooth parametric representation of
  stellar streams with cubic spline interpolation
- 📐 **Curvature analysis**: Compute geometric properties including tangent
  vectors, curvature, principal normals, and arc-length
- 🌌 **Gravitational field fitting**: Match stream curvature to potential models
  with customizable halo and disk components
- ⚡ **JAX-accelerated**: Fast, GPU-compatible computations with automatic
  differentiation and JIT compilation
- 📊 **Likelihood framework**: Bayesian inference for potential parameters using
  curvature-acceleration alignment
- 📈 **Visualization tools**: Built-in plotting methods for tracks, geometry
  vectors, and gravitational fields

## Installation

<!-- [![PyPI version][pypi-version]][pypi-link]
[![PyPI platforms][pypi-platforms]][pypi-link] -->

::::{tab-set}

:::{tab-item} pip

```bash
pip install potamides
```

:::

:::{tab-item} uv

```bash
uv add potamides
```

:::

:::{tab-item} source, via pip

```bash
pip install git+https://github.com/xggs-dev/potamides.git
```

:::

:::{tab-item} building from source

```bash
cd /path/to/parent
git clone https://github.com/xggs-dev/potamides.git
cd potamides
pip install -e .  # editable mode
```

:::

::::

## Quickstart

The `potamides` package provides tools for fitting stellar streams with splines
and inferring the gravitational potential from the stream's curvature and
acceleration. The high-level workflow uses the object-oriented interface of the
`potamides.Track` class. Lower-level functions for spline fitting, etc. can be
found in the `potamides.splinelib` module.

```{code-block} python
>>> import potamides as ptd
>>> import potamides.splinelib as splib
```

We will also assume the following imports for computing and visualization:

```{code-block} python
>>> import functools as ft
>>> import jax
>>> import jax.numpy as jnp
>>> import jax.random as jr
>>> import matplotlib.pyplot as plt
>>> import unxt as u
```

### Fitting a 'Stream'

Let's create a mock stream in 2D (an oval) and fit it with a spline. In
practice, you would replace this with your actual stream data.

```{code-block} python
>>> a, b = 2.0, 1.2  # semi-axes (x, y)
>>> gamma = jnp.linspace(0, 2 * jnp.pi, 10_000)

>>> # Option 1: Simple ellipse
>>> # data = jnp.stack([a * jnp.cos(gamma), b * jnp.sin(gamma)], axis=-1)

>>> # Option 2: Egg-shape (distorted ellipse) - more interesting curvature
>>> k = 0.25  # distortion factor; k=0 gives a perfect ellipse
>>> x = a * jnp.cos(gamma) * (1 + k * jnp.cos(gamma))
>>> y = b * jnp.sin(gamma)
>>> data = jnp.stack([x, y], axis=-1)
```

Then we make the stream track. We choose 25 knots for the spline fit. For
demonstration purposes we'll set the knot position directly from the data.

```{code-block} python
>>> track = ptd.Track(gamma[::400], data[::400])
```

Now we can visualize the 'stream':

```{code-block} python
>>> fig, ax = plt.subplots(figsize=(4, 4), dpi=150)
>>> _gamma = jnp.linspace(track.gamma.min(), track.gamma.max(), num=8)  # for plotting only
>>> _ = track.plot_all(_gamma, ax=ax, show_tangents=False)
```

### Initial basic function for calculating the likelihood

```{code-block} python
>>> params_defaults = {
...    "rs_halo": 16,  # [kpc]
...    "vc_halo": u.Quantity(250, "km/s").ustrip("kpc/Myr"),
...    "q1": 1.0,
...    "q2": 1.0,
...    "q3": 1.0,
...    "phi": 0.0,
...    "Mdisk": 1.2e10,  # [Msun]
...    "origin_x": 0,
...    "origin_y": 0,
...    "origin_z": 0,
...    "rot_z": 0.0,
...    "rot_x": 0.0,
... }
>>> params_statics = {"withdisk": False}

>>> @jax.jit
... def compute_acc_hat(p, /, pos2d):
...    # Positions: 2D -> 3D
...    pos3d = jnp.zeros((len(pos2d), 3))
...    pos3d = pos3d.at[:, :2].set(pos2d)
...    # Accelerations from the potential model
...    p = params_defaults | params_statics | p
...    p["origin"] = jnp.array([p.pop("origin_x", 0), p.pop("origin_y", 0), p.pop("origin_z", 0)])
...    return ptd.compute_accelerations(pos3d, **p)

>>> @ft.partial(jax.vmap, in_axes=(0, None, None))
... @jax.jit
... def compute_ln_likelihood(p, /, pos2d, unit_kappa):
...    unit_acc_xy = compute_acc_hat(p, pos2d)
...    where_straight = jnp.zeros(len(unit_kappa), dtype=bool)
...    return ptd.compute_ln_likelihood(unit_kappa, unit_acc_xy, where_straight) / len(unit_kappa)
```

### Calculating the likelihood

```{code-block} python
>>> ranges = {"q2": (0.1, 2), "phi": (-jnp.pi/2, jnp.pi / 2)}

>>> key, *skeys = jr.split(jr.key(0), num=len(ranges) + 1)
>>> nsamples = 1_000_000
>>> params = {
...     k: jr.uniform(skey, minval=v[0], maxval=v[1], shape=nsamples)
...     for skey, (k, v) in zip(skeys, ranges.items(), strict=True)
... }

>>> gamma = jnp.linspace(-0.95, 0.95, 128)

>>> lnlik_seg = compute_ln_likelihood(params, track(gamma), track.curvature(gamma))
```

Now we can visualize the inference result in 2D. The following creates a scatter
plot showing the likelihood distribution across the parameter space:

```{code-block} python
>>> # Note: This visualization requires all the setup from previous code blocks
>>> # For demonstration purposes, this shows the expected workflow
>>> # In practice, you would run all blocks sequentially in a notebook or script
>>> print("Visualization would show likelihood in (q2, phi) parameter space")
Visualization would show likelihood in (q2, phi) parameter space
```

## Citation

If you use this software in your research, please cite it as:

```bibtex
@software{potamides2024,
  author = {Nibauer, Jacob and Starkman, Nathaniel and Wu, Sirui},
  title = {potamides: Constraining gravitational potentials with stellar stream curvature},
  year = {2024},
  url = {https://github.com/xggs-dev/potamides}
}
```

## Ecosystem

This package builds upon excellent open-source scientific software:

- [JAX][jax]: High-performance numerical computing with automatic
  differentiation
- [galax][galax]: Galactic dynamics in JAX
- [Equinox][equinox]: One-stop JAX library for everything that isn't already in
  core JAX
- [interpax][interpax]: Interpolation library for JAX
- [Astropy][astropy]: Community Python library for astronomy
- [unxt][unxt]: Unitful quantities for JAX

<!-- LINKS -->

[astropy]: https://www.astropy.org/
[equinox]: https://docs.kidger.site/equinox/
[galax]: https://github.com/GalacticDynamics/galax
[interpax]: https://github.com/f0uriest/interpax
[jax]: https://github.com/google/jax
[unxt]: https://github.com/GalacticDynamics/unxt

## Indices and tables

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`
