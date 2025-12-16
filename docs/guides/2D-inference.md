# 2D inference

```{code-block} python
>>> import corner
>>> import numpy as np
>>> import jax.numpy as jnp
>>> import jax.random as jr
>>> import potamides as ptd

>>> # Create a simple example track (same as 1D-inference)
>>> import interpax
>>> from potamides import splinelib as splib
>>> from xmmutablemap import ImmutableMap
>>> t = np.linspace(-np.pi, np.pi, 50)
>>> x_cent = 20 * np.cos(t)
>>> y_cent = 20 * np.sin(t)
>>> xy_centered = jnp.stack([jnp.array(x_cent), jnp.array(y_cent)], axis=1)
>>> fid_gamma, fid_knots = splib.make_increasing_gamma_from_data(xy_centered)
>>> fiducial_spline = interpax.Interpolator1D(fid_gamma, fid_knots, method="cubic2")
>>> num_knots = 10
>>> knots = splib.optimize_spline_knots(
...     splib.default_cost_fn,
...     fid_knots,
...     fid_gamma,
...     cost_args=(jnp.linspace(fid_gamma.min(), fid_gamma.max(), num=128), fiducial_spline(jnp.linspace(fid_gamma.min(), fid_gamma.max(), num=128))),
...     cost_kwargs=ImmutableMap({"concavity_weight": 1e12}),
... )
>>> opt_gamma, opt_knots = splib.new_gamma_knots_from_spline(
...     interpax.Interpolator1D(fid_gamma, knots, method="cubic2"), nknots=num_knots
... )
>>> track = ptd.Track(opt_gamma, opt_knots)

>>> # Define compute_ln_likelihood (simplified version for demo)
>>> import jax
>>> def compute_ln_likelihood_scalar(params_dict, pos2d, unit_curvature, where_straight):
...     return jnp.sum(unit_curvature)  # Simplified for demo
>>> compute_ln_likelihood = jax.vmap(
...     compute_ln_likelihood_scalar, in_axes=(0, None, None, None)
... )

>>> ranges = {
...     "q1": (0.1, 1),
...     "phi": (-np.pi/2, np.pi / 2),
... }

>>> key = jr.key(0)
>>> skeys = jr.split(key, num=len(ranges))
>>> nsamples = 1_000_000
>>> params = {
...     k: jr.uniform(skey, minval=v[0], maxval=v[1], shape=nsamples)
...     for skey, (k, v) in zip(skeys, ranges.items(), strict=True)
... }
>>> params  # doctest: +ELLIPSIS
{...}

>>> gamma = jnp.linspace(-0.95, 0.95, 128)

>>> lnlik_seg = compute_ln_likelihood(
...     params, track(gamma), track.curvature(gamma), None
... )

>>> import matplotlib.pyplot as plt

>>> hist2d_kw = {
...     "bins": 20,
...     "color": "purple",
...     "levels": [0.68, 0.95, 0.997],
...     "plot_density": True,
...     "plot_contours": True,
...     "plot_datapoints": False,
... }

>>> fig, ax = plt.subplots(figsize=(4, 4))
>>> _ = corner.hist2d(
...     params["q1"],
...     params["phi"] * 180 / jnp.pi,  # convert to degrees
...     weights=np.exp(lnlik_seg - lnlik_seg.max()),
...     **hist2d_kw,
... )
>>> _ = ax.set_xlabel('q')
>>> _ = ax.set_ylabel('phi')

```
