---
title: "Potamides: A Python package for stream curvature analysis"
tags:
  - Python
  - astronomy
  - stellar stream
authors:
  - name: Sirui Wu
    orcid: 0009-0003-4675-3622
    equal-contrib: true
    affiliation: 1
  - name: Nathaniel Starkman
    orcid: 0000-0003-3954-3291
    equal-contrib: true
    affiliation: 2
  - name: Jacob Nibauer
    orcid: 0000-0001-8042-5794
    corresponding: true
    affiliation: 3
  - name: Sarah Pearson
    orcid: 0000-0003-0256-5446
    corresponding: true
    affiliation: 1

affiliations:
  - name: Niels Bohr Institute, University of Copenhagen, Denmark
    index: 1
    ror: 035b05819
  - name:
      Brinson Prize Fellow at Kavli Institute for Astrophysics and Space
      Research, Massachusetts Institute of Technology, USA
    index: 2
    ror: 042nb2s44
  - name: Department of Physics, Princeton University, USA
    index: 3
    ror: 00hx57361
date: 7 Dec 2025
bibliography: paper.bib
---

# Summary

`Potamides` is a Python package for constraining galactic gravitational
potentials from the geometry of stellar streams. Stellar streams—elongated
structures of stars tidally stripped from globular clusters or dwarf
galaxies—trace the gravitational field of their host galaxy, and their local
curvature encodes information about dark matter halo properties (flattening and
orientation), baryonic components (disk parameters and galactic center
position), and mass distribution. `Potamides` implements the
curvature–gravitational-acceleration alignment likelihood framework introduced
by @Nibauer:2023, coupling JAX [@jax]-accelerated spline representations of
stream tracks with fast evaluations of gravitational accelerations from flexible
halo and disk potentials. This enables an end-to-end workflow: building smooth
stream tracks from observed positions; computing tangents, principal normals,
and scalar curvature; evaluating gravitational accelerations under candidate
potentials; and combining segment-wise likelihoods across multiple streams with
density-based weighting.

The package integrates with `unxt` [@unxt] for JAX-compatible units and the
`galax` library [@galax] for potential evaluations. JAX's just-in-time (JIT)
compilation and vectorization enable rapid likelihood evaluation across large
parameter spaces—a critical capability for Bayesian inference and parameter
exploration with modern samplers. By focusing on stream curvature rather than
orbit integration, `Potamides` provides a complementary approach to traditional
galactic dynamics tools. The package includes comprehensive tests, documented
examples, and is distributed under the MIT license, making high-performance
curvature-based inference readily accessible to the astronomical community.

# Statement of need

Constraining the shape and structure of dark matter halos is essential for
understanding galaxy formation and testing cosmological models. Stellar
streams—tidal debris from disrupted satellites (dwarf galaxies or globular
clusters)—provide powerful tracers of the galactic gravitational potential
because their morphology encodes the host halo's properties [@Bonaca:2014]. The
curvature-based inference method [@Nibauer:2023] offers a novel approach: it
compares the local curvature of observed streams with predicted gravitational
accelerations, enabling robust constraints on halo flattening and orientation.
However, until now, this methodology lacked a well-documented, accessible, and
high-performance software implementation.

`Potamides` fills this gap by providing the first open-source, production-ready
implementation of the curvature-based inference framework. The package addresses
three critical needs in modern extragalactic dynamics research:

**1. Accessible implementation of a novel method.** While the original work
demonstrated the power of curvature-based inference, it did not provide a
standardized software package for community adoption. `Potamides` makes this
methodology readily available to researchers, lowering the barrier to entry and
enabling reproducible science. The well-documented API and examples allow users
to apply curvature-based constraints without reimplementing the complex
likelihood framework.

**2. Scalability for upcoming survey data.** Next-generation imaging surveys
will discover hundreds to thousands of stellar streams in nearby galaxies
[@Mateu:2023]. These datasets will enable population-level studies of halo
properties across galaxy types and environments. `Potamides` is designed to
handle this data volume efficiently.

**3. High-performance parameter space exploration.** Bayesian inference for halo
parameters requires evaluating likelihoods across large, multi-dimensional
parameter spaces. Traditional Python implementations would be prohibitively slow
for such exploration. `Potamides` leverages JAX's just-in-time (JIT) compilation
and automatic vectorization to achieve 10-100x speedups compared to numpy-based
alternatives, with optional GPU acceleration for even larger performance gains.
This efficiency is critical for modern sampling methods (e.g., Hamiltonian Monte
Carlo, nested sampling) that require thousands to millions of likelihood
evaluations.

# Acknowledgements

This work was supported by a research grant (VIL53081) from VILLUM FONDEN. This
work was also co-funded by the European Union (ERC, BeyondSTREAMS, 101115754)
grant. Views and opinions expressed are however those of the author(s) only and
do not necessarily reflect those of the European Union or the European Research
Council. Neither the European Union nor the granting authority can be held
responsible for them. The Tycho supercomputer hosted at the SCIENCE HPC center
at the University of Copenhagen was used for supporting this work.

# References
