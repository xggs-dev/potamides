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

This quick example demonstrates the basic workflow for fitting a stellar stream
with a gravitational potential model. We'll use data from
[Nibauer et al. (2023)](https://arxiv.org/abs/2303.17406) to illustrate the
method.

For a complete interactive tutorial, see the
[Stream Fitting Guide](guides/stream_fitting.md).

### Step 0: Import Required Libraries

First, enable JAX 64-bit precision and import the necessary packages:

```{code-block} python
>>> import jax
>>> jax.config.update("jax_enable_x64", True)

>>> import numpy as np
>>> import matplotlib.pyplot as plt
>>> import jax.numpy as jnp
>>> import jax.random as jr
>>> import unxt as u

>>> import potamides as ptd
>>> from potamides import splinelib
```

### Step 1: Prepare Stream Data

Load or define your stream coordinates. Here we use control points extracted
from the literature:

```{code-block} python
>>> # Example: manually extracted from Nibauer et al. (2023), Figure 5
>>> xy = np.array([
...     [-1.02940125e-02, -1.09604831e+01],
...     [-9.90913652e+00, -8.79524192e+00],
...     [-1.66558337e+01,  5.41573739e-03],
...     [-2.01397838e+01,  9.92361722e+00],
...     [-1.53839153e+01,  2.16578274e+01],
...     [-4.32375611e+00,  1.39048671e+01],
...     [ 4.85617605e+00, -2.04123731e-01],
...     [ 1.03309549e+01, -1.32654172e+01],
...     [ 1.28194907e+01, -2.02500662e+01]
... ])
```

### Step 2: Create Spline Track

Parameterize the stream using arc-length and construct a `Track` object:

```{code-block} python
>>> def make_gamma_from_data(data):
...     """Compute normalized arc-length parameter gamma ∈ [-1, 1]"""
...     s = splinelib.point_to_point_arclength(data)
...     s = jnp.concat((jnp.array([0]), s))
...     s_min = s.min()
...     gamma = 2 * (s - s_min) / (s.max() - s_min) - 1
...     return gamma

>>> gamma = make_gamma_from_data(xy)
>>> track = ptd.Track(gamma, xy)
```

Visualize the track:

```{code-block} python
>>> fig, ax = plt.subplots(figsize=(5, 5), dpi=150)
>>> plt.plot(0, 0, 'r*', markersize=12, label='Galactic center')
>>> plt.plot(track.knots[:, 0], track.knots[:, 1], 'o', color='orange', markersize=6, label='Knots')
>>> plot_sparse_gamma = jnp.linspace(-1, 1, num=30)
>>> track.plot_all(plot_sparse_gamma, ax=ax, show_tangents=False)
>>> ax.set_xlabel("X (kpc)")
>>> ax.set_ylabel("Y (kpc)")
>>> ax.set_aspect('equal')
>>> ax.legend()
```

### Step 3: Define Potential Model

Set up a triaxial NFW halo potential with parameters:

```{code-block} python
>>> params_defaults = {
...     # Halo structure (typically fixed)
...     "rs_halo": 16,
...     "vc_halo": u.Quantity(250, "km/s").ustrip("kpc/Myr"),
...     # Halo shape (q2 is commonly fitted)
...     "q1": 1.0,
...     "q2": 1.0,  # ← y-axis flattening parameter
...     "q3": 1.0,
...     # Halo orientation
...     "phi": 0.0,  # ← long-axis orientation angle
...     # Halo center position
...     "origin_x": 0, "origin_y": 0, "origin_z": 0,
...     # Additional components
...     "Mdisk": 5e12,
...     "rot_z": 0.0, "rot_x": 0.0,
... }
>>> params_statics = {"withdisk": False}

>>> @jax.jit
... def compute_acc_hat(params, pos2d):
...     """Compute unit acceleration vectors."""
...     pos3d = jnp.zeros((len(pos2d), 3))
...     pos3d = pos3d.at[:, :2].set(pos2d)
...     merged = params_defaults | params_statics | params
...     merged["origin"] = jnp.array([
...         merged.pop("origin_x", 0),
...         merged.pop("origin_y", 0),
...         merged.pop("origin_z", 0),
...     ])
...     return ptd.compute_accelerations(pos3d, **merged)

>>> @jax.jit
... def compute_ln_likelihood_scalar(params, pos2d, unit_curvature):
...     """Compute log-likelihood for a parameter set."""
...     unit_acc_xy = compute_acc_hat(params, pos2d)
...     where_straight = jnp.zeros(len(unit_curvature), dtype=bool)
...     lnlik = ptd.compute_ln_likelihood(
...         unit_curvature, unit_acc_xy, where_straight=where_straight
...     )
...     return lnlik - jnp.log(len(unit_curvature))

>>> compute_ln_likelihood = jax.vmap(
...     compute_ln_likelihood_scalar, in_axes=(0, None, None)
... )
```

### Step 4: Sample Parameter Space

Scan the q2 parameter (y-axis flattening) to find the best fit:

```{code-block} python
>>> ranges = {"q2": (0.1, 2.0)}
>>> key = jr.key(0)
>>> skeys = jr.split(key, num=len(ranges))
>>> nsamples = 1_000

>>> params = {
...     k: jr.uniform(skey, minval=v[0], maxval=v[1], shape=(nsamples,))
...     for skey, (k, v) in zip(skeys, ranges.items(), strict=True)
... }

>>> lnlik_seg = compute_ln_likelihood(
...     params,
...     track(gamma),
...     track.curvature(gamma),
... )
```

### Step 5: Visualize Results

Plot the relative likelihood as a function of q2:

```{code-block} python
>>> q = np.array(params['q2'])
>>> lnlik_seg_np = np.array(lnlik_seg)
>>> idx = np.argsort(q)
>>> q_sorted = q[idx]
>>> lnlik_sorted = lnlik_seg_np[idx]

>>> fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
>>> plt.plot(q_sorted, np.exp(lnlik_sorted - lnlik_sorted.max()),
...          'c-', linewidth=2, label='Relative likelihood')
>>> plt.xlim(0.0, 2.0)
>>> plt.ylim(0.0, 1.05)
>>> plt.xlabel(r"$q_2$ (y-axis flattening)")
>>> plt.ylabel(r"$\mathcal{L}/\mathcal{L}_{\max}$")
>>> plt.legend()
>>> plt.grid(alpha=0.3)

>>> # Find best-fit parameter
>>> idx_max = np.argmax(lnlik_sorted)
>>> q_best = q_sorted[idx_max]
>>> print(f"Best-fit q2 = {q_best:.3f}")  # doctest: +SKIP
Best-fit q2 = 0.725
```

### Next Steps

This quickstart covered single-parameter fitting. For more advanced analyses:

- **Multi-parameter fitting**: Scan q1, q2, q3, phi, and origin simultaneously
- **Bayesian inference**: Use MCMC (numpyro, emcee) for full posterior
  distributions
- **Real data**: Apply to observed streams from Gaia or other surveys

See the [Guides](guides/begin.md) for detailed tutorials and the
[API Reference](api/index.md) for complete documentation.

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
