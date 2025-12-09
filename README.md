# Potamides

[![Actions Status][actions-badge]][actions-link]
[![Documentation Status][rtd-badge]][rtd-link]

[![PyPI version][pypi-version]][pypi-link]
[![Conda-Forge][conda-badge]][conda-link]
[![PyPI platforms][pypi-platforms]][pypi-link]

[![GitHub Discussion][github-discussions-badge]][github-discussions-link]

<!-- SPHINX-START -->

**potamides** is a Python package for constraining gravitational potentials
using stellar stream curvature analysis. The name is inspired by Greek ποταμίδες
("potamídes", meaning "river streams"), with the initial "P" representing
$\Phi$, the conventional symbol for gravitational potential in astronomy.

### Key Features

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

### Using uv (recommended)

```bash
uv add potamides
```

### Using pip

```bash
pip install potamides
```

### From source

```bash
git clone https://github.com/xggs-dev/potamides.git
cd potamides
uv pip install -e .
```

### Requirements

- Python ≥ 3.11
- JAX ≥ 0.5.3
- For GPU support, install JAX with CUDA support separately
- See `pyproject.toml` for full dependency list

</br>

## About

Constraining the three-dimensional shape and structure of dark matter halos is
essential for understanding galaxy formation and testing cosmological models.
Stellar streams—tidal debris from disrupted satellites—provide powerful tracers
of the galactic gravitational potential because their morphology directly
reflects the host halo's properties [@Koposov:2010; @Bonaca:2014]. The
curvature-based inference method introduced by @Nibauer:2023 offers a novel
approach: instead of integrating orbits, it compares the local curvature of
observed streams with predicted gravitational accelerations, enabling robust
constraints on halo flattening, orientation, and baryonic structure. However,
until now, this methodology lacked a well-documented, accessible, and
high-performance software implementation.

`Potamides` fills this gap by providing the first open-source, production-ready
implementation of the Nibauer et al. (2023) framework. The package addresses
three critical needs in modern extragalactic dynamics research:

**1. Accessible implementation of a novel method.** While the original
@Nibauer:2023 work demonstrated the power of curvature-based inference, it did
not provide a standardized software package for community adoption. `Potamides`
makes this methodology readily available to researchers, lowering the barrier to
entry and enabling reproducible science. The well-documented API and examples
allow users to apply curvature-based constraints without reimplementing the
complex likelihood framework.

**2. Scalability for upcoming survey data.** Next-generation imaging surveys—
including the Vera C. Rubin Observatory's Legacy Survey of Space and Time
(LSST), Euclid, and the Nancy Grace Roman Space Telescope—will discover hundreds
to thousands of stellar streams in nearby galaxies [@Mateu:2023]. These datasets
will enable population-level studies of halo properties across galaxy types and
environments. `Potamides` is designed to handle this data volume efficiently:
its JAX backend supports batch processing of multiple streams, and the
curvature-based approach is computationally lighter than traditional N-body or
orbit-fitting methods.

**3. High-performance parameter space exploration.** Bayesian inference for halo
parameters requires evaluating likelihoods across large, multi-dimensional
parameter spaces (typically 5–10 parameters including halo axis ratios,
orientation angles, baryonic center coordinates, and disk properties).
Traditional Python implementations would be prohibitively slow for such
exploration. `Potamides` leverages JAX's just-in-time (JIT) compilation and
automatic vectorization to achieve 10–100x speedups compared to NumPy-based
alternatives, with optional GPU acceleration for even larger performance gains.
This efficiency is critical for modern sampling methods (e.g., Hamiltonian Monte
Carlo, nested sampling) that require thousands to millions of likelihood
evaluations.

`Potamides` is designed for both expert researchers conducting detailed halo
studies and students learning about stellar streams and gravitational dynamics.
It integrates seamlessly with the Python astronomy ecosystem (`Astropy`,
`galax`) and follows best practices for scientific software (comprehensive
tests, documented examples, version control, continuous integration). The
package has already been applied to constrain halo properties in several
extragalactic stream systems, demonstrating its scientific utility. By making
curvature-based potential inference accessible, efficient, and reproducible,
`Potamides` enables the community to fully exploit the diagnostic power of
stellar streams in the era of large-scale surveys.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file
for details.

## Acknowledgments

This package builds upon excellent open-source scientific software:

- **[JAX](https://github.com/google/jax)**: High-performance numerical computing
  with automatic differentiation
- **[galax](https://github.com/GalacticDynamics/galax)**: Galactic dynamics in
  JAX
- **[unxt](https://github.com/GalacticDynamics/unxt)**: Unitful quantities for
  JAX
- **[interpax](https://github.com/f0uriest/interpax)**: Interpolation library
  for JAX
- **[Astropy](https://www.astropy.org/)**: Community python library for
  astronomy

---

<!-- prettier-ignore-start -->
[actions-badge]:            https://github.com/wsr1998/potamides/workflows/CI/badge.svg
[actions-link]:             https://github.com/wsr1998/potamides/actions
[conda-badge]:              https://img.shields.io/conda/vn/conda-forge/potamides
[conda-link]:               https://github.com/conda-forge/potamides-feedstock
[github-discussions-badge]: https://img.shields.io/static/v1?label=Discussions&message=Ask&color=blue&logo=github
[github-discussions-link]:  https://github.com/wsr1998/potamides/discussions
[pypi-link]:                https://pypi.org/project/potamides/
[pypi-platforms]:           https://img.shields.io/pypi/pyversions/potamides
[pypi-version]:             https://img.shields.io/pypi/v/potamides
[rtd-badge]:                https://readthedocs.org/projects/potamides/badge/?version=latest
[rtd-link]:                 https://potamides.readthedocs.io/en/latest/?badge=latest

<!-- prettier-ignore-end -->
