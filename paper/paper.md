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

Potamides is a Python package for inferring galactic gravitational potentials
from the geometry of stellar streams. Stellar streams—elongated structures
formed by stars tidally stripped from globular clusters or dwarf galaxies—trace
the gravitational field of their host galaxy. Their local curvature contains
information about the underlying mass distribution, enabling direct inference of
dark matter halo shape and orientation, baryonic components such as the disk,
and the position of the galactic center.

`Potamides` implements the curvature–gravitational-acceleration alignment
likelihood framework introduced by [@Nibauer:2023]. The package couples
JAX-accelerated spline representations of stream tracks with fast evaluations of
gravitational accelerations from flexible halo and disk potentials. This
supports an end-to-end workflow: constructing smooth stream tracks from observed
positions; computing tangents, principal normals, and scalar curvature;
evaluating gravitational accelerations under candidate potentials; and combining
segment-wise likelihoods across multiple streams using density-based weighting.

The package integrates with `unxt` [@unxt] for JAX-compatible units and with
`galax` [@galax] for gravitational potential evaluation. JAX’s just-in-time
(JIT) compilation and vectorization enable efficient evaluation of
curvature-based likelihoods, which is essential for Bayesian inference and
parameter exploration with modern samplers. By focusing on stream curvature
rather than explicit orbit integration, `Potamides` provides a complementary
framework for gravitational potential inference in galactic dynamics. The
package includes comprehensive tests and documented examples, making
high-performance, curvature-based inference accessible to the astronomical
community.

# Statement of need

Constraining the shape and structure of dark matter halos is central to
understanding galaxy formation and testing cosmological models. Stellar
streams—tidal debris from disrupted satellites such as dwarf galaxies or
globular clusters—serve as sensitive tracers of the galactic gravitational
potential because their morphology encodes the host halo’s properties
[@Bonaca:2014]. The curvature-based inference method introduced by
[@Nibauer:2023] provides a novel approach by comparing the local curvature of
observed streams with predicted gravitational accelerations, enabling robust
constraints on halo flattening and orientation. Until recently, however, this
methodology lacked a well-documented, accessible, and high-performance software
implementation.

`Potamides` fills this gap by providing an open-source, production-ready
implementation of the curvature-based inference framework. The package addresses
three critical needs in modern extragalactic dynamics research:

**1. Accessible implementation of a curvature-based method.** The
curvature-based inference framework introduced by [@Nibauer:2023] was not
originally accompanied by a standardized software package for community use.
`Potamides` provides a reference implementation developed in collaboration with
the author of the original work. A well-documented API and reproducible examples
enable researchers to apply curvature-based constraints without reimplementing
the underlying likelihood framework.

**2. Scalability for upcoming survey data.** Next-generation imaging surveys are
expected to discover hundreds to thousands of stellar streams in nearby galaxies
[@Mateu:2023], enabling population-level studies of halo properties across
galaxy types and environments. `Potamides` is designed to scale efficiently to
these data volumes and to support analyses involving large numbers of stream
segments and host galaxies.

**3. High-performance parameter space exploration.** Bayesian inference of halo
parameters requires repeated likelihood evaluations over high-dimensional
gravitational potential models. `Potamides` leverages JAX’s just-in-time (JIT)
compilation and automatic vectorization to enable rapid likelihood evaluation,
achieving order-of-magnitude speedups over NumPy-based implementations and
supporting modern sampling methods that require large numbers of model
evaluations.

# Acknowledgements

This work was supported by a research grant (VIL53081) from VILLUM FONDEN. This
work was also co-funded by the European Union (ERC, BeyondSTREAMS, 101115754)
grant. Views and opinions expressed are however those of the author(s) only and
do not necessarily reflect those of the European Union or the European Research
Council. Neither the European Union nor the granting authority can be held
responsible for them. The Tycho supercomputer hosted at the SCIENCE HPC center
at the University of Copenhagen was used for supporting this work.

Support for this work was provided by The Brinson Foundation through a Brinson
Prize Fellowship grant to N.S.

# References
