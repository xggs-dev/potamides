# 1D-inference

```{code-block} python
>>> import potamides as ptd
>>> import unxt as u
>>> import jax
>>> import jax.numpy as jnp
>>> import jax.random as jr

>>> @jax.jit
... def compute_acc_hat(params, pos2d):
...     # Positions: 2D -> 3D
...     pos3d = jnp.zeros((len(pos2d), 3))
...     pos3d = pos3d.at[:, :2].set(pos2d)
...
...     # Accelerations from the potential model
...     params = params_defaults | params_statics | params
...     params["origin"] = jnp.array(
...         [
...             params.pop("origin_x", 0),
...             params.pop("origin_y", 0),
...             params.pop("origin_z", 0),
...         ]
...     )
...     return ptd.compute_accelerations(pos3d, **params)

>>> @jax.jit
... def compute_ln_likelihood_scalar(params, pos2d, unit_curvature, where_straight=None):
...     unit_acc_xy = compute_acc_hat(params, pos2d)
...     where_straight = (
...         where_straight
...         if where_straight is not None
...         else jnp.zeros(len(unit_curvature), dtype=bool)
...     )
...     return ptd.compute_ln_likelihood(
...         unit_curvature, unit_acc_xy, where_straight=where_straight
...     ) / len(unit_curvature)

>>> # parameters for the halo
>>> params_defaults = {
...     "rs_halo": 16,  # [kpc]
...     "vc_halo": u.Quantity(250, "km/s").ustrip("kpc/Myr"),
...     "q1": 1.0,
...     "q2": 1.0,
...     "q3": 1.0,
...     "phi": 0.0,
...     "Mdisk": 1.2e10,  # [Msun]
...     "origin_x": 0,
...     "origin_y": 0,
...     "origin_z": 0,
...     "rot_z": 0.0,
...     "rot_x": 0.0,
... }
>>> params_statics = {"withdisk": False}


>>> compute_ln_likelihood = jax.vmap(
...     compute_ln_likelihood_scalar, in_axes=(0, None, None, None)
... )

>>> ranges = {
...     "q1": (0.1, 2),
... }

>>> key = jr.key(0)
>>> skeys = jr.split(key, num=len(ranges))
>>> nsamples = 1_000
>>> params = {
...     k: jr.uniform(skey, minval=v[0], maxval=v[1], shape=nsamples)
...     for skey, (k, v) in zip(skeys, ranges.items(), strict=True)
... }

>>> # Create a simple example track for demonstration
>>> # In practice, you would create this from your stream data
>>> import numpy as np
>>> t = np.linspace(-np.pi, np.pi, 50)
>>> x_cent = 20 * np.cos(t)
>>> y_cent = 20 * np.sin(t)
>>> xy_centered = jnp.stack([jnp.array(x_cent), jnp.array(y_cent)], axis=1)

>>> from potamides import splinelib as splib
>>> import interpax
>>> fid_gamma, fid_knots = splib.make_increasing_gamma_from_data(xy_centered)
>>> fiducial_spline = interpax.Interpolator1D(fid_gamma, fid_knots, method="cubic2")
>>> from xmmutablemap import ImmutableMap
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

>>> gamma = jnp.linspace(-0.95, 0.95, 128)

>>> lnlik_seg = compute_ln_likelihood(
...     params, track(gamma), track.curvature(gamma), None
... )

```
