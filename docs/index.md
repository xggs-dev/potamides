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

guides/stream_fitting.md
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
with a gravitational potential model. We'll use StreamB data from Figure 5
(second panel) of [Nibauer et al. (2023)](https://arxiv.org/abs/2303.17406) to
illustrate the method.

### Workflow Overview

The analysis consists of five main steps:

1. **Prepare control points** - Get ordered (x, y) coordinates of the stream
   (user-provided data)
2. **Create spline track** - Fit a parametric spline representation to the
   control points
3. **Define potential model** - Set up the trial halo potential parameters
4. **Sample parameter space** - Compute the likelihood for parameter samples
5. **Visualize results** - Plot the likelihood distribution and best-fit
   parameters

For a complete interactive tutorial, see the
[Stream Fitting Guide](guides/stream_fitting.md).

### Step 0: Import Required Libraries

First, enable JAX 64-bit precision and import the necessary packages:

```{code-cell} ipython3
import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax.random as jr
import unxt as u

import potamides as ptd
from potamides import splinelib
```

### Step 1: Prepare Stream Data

Load or define your stream coordinates. Here we use control points manually
extracted from StreamB in
[Nibauer et al. (2023)](https://arxiv.org/abs/2303.17406) (Figure 5, second
panel):

```{code-cell} ipython3
# Example: manually extracted from Nibauer et al. (2023), Figure 5, second panel
xy = np.array([[-18.23818192,   7.7713813 ],
               [-23.20527332,  13.30501798],
               [-25.68881901,  17.85818509],
               [-26.82711079,  22.51483327],
               [-26.51666758,  26.81189831],
               [-20.04832301,  26.39824245],
               [-17.02861664,  24.77713165],
               [ -9.74451624,  19.97321842],
               [ -4.03009244,  14.80896887]])

print(f"Stream contains {len(xy)} control points")
```

### Step 2: Create Spline Track

Parameterize the stream using arc-length and construct a `Track` object.

Note: In this example, we directly use the control points to construct the
spline without further optimization, as the reference points have already been
carefully selected. For automatic knot optimization, see
`splinelib.optimize_spline_knots` and related functions discussed in the
[Stream Fitting Guide](guides/stream_fitting.md).

```{code-cell} ipython3
def make_gamma_from_data(data):
    """Compute normalized arc-length parameter gamma ∈ [-1, 1]"""
    s = splinelib.point_to_point_arclength(data)
    s = jnp.concat((jnp.array([0]), s))
    s_min = s.min()
    gamma = 2 * (s - s_min) / (s.max() - s_min) - 1
    return gamma

gamma = make_gamma_from_data(xy)
track = ptd.Track(gamma, xy)

print(f"Track created with {len(gamma)} knots")
print(f"Gamma range: [{gamma.min():.3f}, {gamma.max():.3f}]")
```

Visualize the track:

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(5, 5), dpi=150)
plt.plot(0, 0, 'r*', markersize=12, label='Galactic center')
plot_sparse_gamma = jnp.linspace(-1, 1, num=30)
track.plot_all(plot_sparse_gamma, ax=ax, show_tangents=False)
ax.set_xlabel("X (kpc)")
ax.set_ylabel("Y (kpc)")
ax.set_xlim(-50, 50)
ax.set_ylim(-50, 50)
ax.set_aspect('equal')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()
```

### Step 3: Define Potential Model

Set up a triaxial NFW halo potential with parameters.

**Important notes on configuration**:

- The parameters shown below are **default values for illustration** — in
  practice, you'll fit some of these (like q2, phi, origin) while keeping others
  fixed
- **Galactic center**: The default assumes the halo is centered at `(0, 0, 0)`.
  If your stream data uses a different coordinate system or the halo is
  off-center, adjust `origin_x`, `origin_y`, `origin_z` accordingly
- **Disk component**: Set `withdisk=False` (default) to use halo-only potential.
  Set to `True` to include a Miyamoto-Nagai disk with mass `Mdisk`

```{code-cell} ipython3
params_defaults = {
    # Halo structure (typically fixed)
    "rs_halo": 16,
    "vc_halo": u.Quantity(250, "km/s").ustrip("kpc/Myr"),
    # Halo shape (q2 is commonly fitted)
    "q1": 1.0,
    "q2": 1.0,  # ← y-axis flattening parameter
    "q3": 1.0,
    # Halo orientation
    "phi": 0.0,  # ← long-axis orientation angle
    # Halo center position (default: galactic center)
    "origin_x": 0, "origin_y": 0, "origin_z": 0,
    # Additional components
    "Mdisk": 5e12,
    "rot_z": 0.0, "rot_x": 0.0,
}
params_statics = {"withdisk": False}  # Halo-only potential (default)

@jax.jit
def compute_acc_hat(params, pos2d):
    """Compute unit acceleration vectors."""
    pos3d = jnp.zeros((len(pos2d), 3))
    pos3d = pos3d.at[:, :2].set(pos2d)
    merged = params_defaults | params_statics | params
    merged["origin"] = jnp.array([
        merged.pop("origin_x", 0),
        merged.pop("origin_y", 0),
        merged.pop("origin_z", 0),
    ])
    return ptd.compute_accelerations(pos3d, **merged)

@jax.jit
def compute_ln_likelihood_scalar(params, pos2d, unit_curvature):
    """Compute log-likelihood for a parameter set."""
    unit_acc_xy = compute_acc_hat(params, pos2d)
    where_straight = jnp.zeros(len(unit_curvature), dtype=bool)
    lnlik = ptd.compute_ln_likelihood(
        unit_curvature, unit_acc_xy, where_straight=where_straight
    )
    return lnlik - jnp.log(len(unit_curvature))

compute_ln_likelihood = jax.vmap(
    compute_ln_likelihood_scalar, in_axes=(0, None, None)
)

print("✓ Likelihood functions defined and JIT compiled")
```

### Step 4: Sample Parameter Space

**About the mock stream**: StreamB was generated with a gravitational potential
having **q2 = 1.0** and **phi = 0**. Our goal is to recover these parameters
from the stream's curvature.

**1D inference example**: In this demonstration, we perform a simplified
1-parameter fit by scanning only q2 while keeping all other parameters fixed.
This illustrates the basic method before moving to multi-parameter fitting (see
[Guides](guides/2D-inference.md)).

**Understanding q2** (following
[Nibauer et al. 2023](https://arxiv.org/abs/2303.17406) convention):

- **q2 = 1**: Spherical halo in the y-direction
- **q2 < 1**: Flattened (oblate) halo, where the y-axis is the short axis
- **q2 > 1**: Prolate (elongated) halo, where the y-axis is stretched

**Parameter range**: We scan q2 ∈ [0.1, 2.0] to reproduce the original Figure 5
from the paper. **Note: In later stream curvature studies, q2 is often redefined
as the short-to-long axis ratio, which restricts values to (0, 1].**

**Commonly fitted parameters in stream analysis**:

- **q1, q2, q3**: Halo axis ratios → constrains dark matter halo shape
- **phi**: Long-axis orientation → determines halo alignment
- **origin_x, origin_y, origin_z**: Halo center position → critical for
  off-center streams

```{code-cell} ipython3
ranges = {"q2": (0.1, 2.0)}
key = jr.key(0)
skeys = jr.split(key, num=len(ranges))
nsamples = 1_000

params = {
    k: jr.uniform(skey, minval=v[0], maxval=v[1], shape=(nsamples,))
    for skey, (k, v) in zip(skeys, ranges.items(), strict=True)
}

print(f"Sampling {nsamples} parameter values for q2 in [{ranges['q2'][0]}, {ranges['q2'][1]}]")

lnlik_seg = compute_ln_likelihood(
    params,
    track(gamma),
    track.curvature(gamma),
)

print(f"Likelihood calculation complete")
print(f"Log-likelihood range: [{jnp.min(lnlik_seg):.3f}, {jnp.max(lnlik_seg):.3f}]")
```

### Step 5: Visualize Results

Plot the relative likelihood as a function of q2 and compare with the true
value:

```{code-cell} ipython3
q = np.array(params['q2'])
lnlik_seg_np = np.array(lnlik_seg)
idx = np.argsort(q)
q_sorted = q[idx]
lnlik_sorted = lnlik_seg_np[idx]

fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
plt.plot(q_sorted, np.exp(lnlik_sorted - lnlik_sorted.max()),
         'c-', linewidth=2, label='Relative likelihood')
plt.vlines(1.0,0,1.1,'r', label='True value (q2=1.0)')
plt.xlim(0.0, 2.0)
plt.ylim(0.0, 1.05)
plt.xlabel(r"$q_2$ (y-axis flattening)", fontsize=14)
plt.ylabel(r"$\mathcal{L}/\mathcal{L}_{\max}$", fontsize=14)
plt.xticks([0.5, 1.0, 1.5])
plt.yticks([0.0, 0.5, 1.0])
plt.tick_params(axis="both", which="major", direction="in",
                top=True, right=True, length=6, width=1.2)
plt.tick_params(axis="both", which="minor", direction="in",
                top=True, right=True, length=3, width=1.0)
plt.minorticks_on()
plt.legend(fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Find best-fit parameter
idx_max = np.argmax(lnlik_sorted)
q_best = q_sorted[idx_max]

```

### Next Steps

This quickstart covered single-parameter fitting. For more advanced analyses:

- [Stream Fitting Guide](guides/stream_fitting.md) - Learn three approaches to
  building Track objects
- [2D Inference Guide](guides/2D-inference.md) - Advanced parameter fitting
  techniques
- [API Reference](api/index.md) - Complete documentation of all functions and
  classes

## Citation

If you use this software in your research, please cite it as:

```bibtex
@software{potamides2024,
  author = {Wu, Sirui and Starkman, Nathaniel and Nibauer, Jacob and Pearson, Sarah},
  title = {Potamides: A Python package for stream curvature analysis},
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
