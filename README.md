# potamides

[![Actions Status][actions-badge]][actions-link]
[![Documentation Status][rtd-badge]][rtd-link]

[![PyPI version][pypi-version]][pypi-link]
[![PyPI platforms][pypi-platforms]][pypi-link]

[![GitHub Discussion][github-discussions-badge]][github-discussions-link]

<!-- SPHINX-START -->

## About

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

### Using pip (recommended)

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

- Python >= 3.11
- JAX >= 0.5.3
- For GPU support, install JAX with CUDA support separately
- See `pyproject.toml` for full dependency list

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file
for details.

## Acknowledgments

This package builds upon excellent open-source scientific software:

- **[JAX](https://github.com/google/jax)**: High-performance numerical computing
  with automatic differentiation
- **[galax](https://github.com/GalacticDynamics/galax)**: Galactic dynamics in
  JAX
- **[interpax](https://github.com/f0uriest/interpax)**: Interpolation library
  for JAX
- **[Astropy](https://www.astropy.org/)**: Community Python library for
  astronomy
- **[unxt](https://github.com/GalacticDynamics/unxt)**: Unitful quantities for
  JAX

---

<!-- prettier-ignore-start -->
[actions-badge]:            https://github.com/xggs-dev/potamides/workflows/CI/badge.svg
[actions-link]:             https://github.com/xggs-dev/potamides/actions
[github-discussions-badge]: https://img.shields.io/static/v1?label=Discussions&message=Ask&color=blue&logo=github
[github-discussions-link]:  https://github.com/xggs-dev/potamides/discussions
[pypi-link]:                https://pypi.org/project/potamides/
[pypi-platforms]:           https://img.shields.io/pypi/pyversions/potamides
[pypi-version]:             https://img.shields.io/pypi/v/potamides
[rtd-badge]:                https://readthedocs.org/projects/potamides/badge/?version=latest
[rtd-link]:                 https://potamides.readthedocs.io/en/latest/?badge=latest

<!-- prettier-ignore-end -->
