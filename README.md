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

## Contributing

potamides is made for its users, so contributions of all kinds are welcome and
encouraged. You do not need to write code to help: reporting a bug, suggesting a
feature, improving the documentation, or sharing a scientific use case are all
valuable.

See [CONTRIBUTING.md](CONTRIBUTING.md) for a full guide, including how to set up
a development environment and run the tests. The easiest ways to get involved
are through GitHub:

- **Found a bug or have a feature request?** Open an
  [issue](https://github.com/xggs-dev/potamides/issues). Please include enough
  detail (a minimal example, error messages, and your environment) for us to
  reproduce the problem.
- **Have a question or an idea to discuss?** Start a thread in
  [Discussions](https://github.com/xggs-dev/potamides/discussions).
- **Want to contribute code or documentation?** Fork the repository, create a
  branch for your change, and open a
  [pull request](https://github.com/xggs-dev/potamides/pulls). If you are new to
  this workflow, GitHub's
  [guide to contributing to projects](https://docs.github.com/en/get-started/exploring-projects-on-github/contributing-to-a-project)
  is a good place to start, and we are happy to help you through the process.

By contributing, you agree to abide by our
[Code of Conduct](CODE_OF_CONDUCT.md).

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

## AI Usage Disclosure

Portions of this codebase (including tests and documentation) were refactored
and generated with the assistance of Language Models. All AI contributions have
been and will continue to be reviewed and verified by the human maintainers.

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
