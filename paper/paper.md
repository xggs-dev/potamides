---
title: "Potamides: JAX tools for curvature-based inference from stellar streams"
tags:
  - Python
  - astronomy
  - stellar streams
  - galactic dynamics
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

`Potamides` is a Python package for inferring the mass distribution of galaxies
from the projected shapes of stellar streams in imaging data. Stellar streams
are elongated structures produced when star clusters or dwarf galaxies are
tidally disrupted by their host. Because their projected tracks carry
information about the host's gravitational field, the local curvature of a
stream can constrain the underlying potential.

The package implements and extends the curvature-based likelihood framework of
[@Nibauer:2023]. Rather than generating a full dynamical realization of a
stellar stream for each trial model, `Potamides` represents observed stream
tracks with JAX-based splines. It evaluates gravitational accelerations in
candidate potentials and compares them directly to the local stream geometry.
This provides a lower-cost inference workflow that complements traditional
forward-modeling approaches. `Potamides` supports the complete analysis pipeline
for a galaxy, from annotating stream ridge-lines to evaluating likelihoods
across many potential models.

# Statement of need

Stellar streams are popular tracers of galactic gravitational potentials and the
dark matter halos that dominate galaxies [@Bonaca:2014]. For external galaxies,
the observed dynamical information is often limited to projected stream
morphology. The curvature-based method of Nibauer et al. [@Nibauer:2023]
addresses this regime by using the local relationship between stream curvature
and gravitational acceleration to directly constrain the potential's geometry
from the projected stream track.

Until now, this method lacked a reusable, high-performance software
implementation intended for community use. `Potamides` fills that gap, serving
as the code base for curvature-based inference. It builds on the original
reference implementation while introducing methodological improvements such as
spline-knot optimization, better handling of locally straight segments, and
joint inference across multiple streams within a common likelihood.

The package is designed for researchers who need to analyze stream systems
end-to-end. Users can estimate a smooth ridge-line from ordered points, extract
curvature observables, and evaluate families of gravitational potentials
efficiently enough to test a wide variety of model assumptions.

# State of the field

`Potamides` occupies a different methodological niche from commonly used
galactic-dynamics packages such as `gala` [@Price-Whelan2017], `galpy`
[@Bovy2015], and `AGAMA` [@Vasiliev2019]. Those libraries provide excellent
tools for orbit integration, action-angle methods, and forward modeling. They
are ideal when the goal is to simulate orbits or generate stream realizations
from explicit physical histories. Recent JAX-based packages like `galax`
[@galax] and `StreamSculptor` [@Nibauer:2025:StreamSculptor] bring
differentiable, GPU-compatible modeling to the ecosystem, but they similarly
focus on classical dynamical calculations and forward simulations.

`Potamides` is designed to complement these tools. It is not a general
galactic-dynamics library. Instead, it treats the projected stream track as the
primary observable and compares it directly with the local acceleration field
implied by a candidate potential. This approach makes fewer assumptions about
progenitor properties and stream formation history than forward models. It is
best suited for problems where fast, direct constraints from morphology are
required, with forward modeling serving as a possible complementary follow-up.

# Software design

The core design goal of `Potamides` is to make curvature-based inference a
practical end-to-end workflow by coupling geometric track modeling, likelihood
evaluation, and visualization within a single framework.

The JAX-first implementation [@jax] enables vectorization, just-in-time
compilation, and hardware portability without requiring separate code paths for
prototyping and production. `Potamides` uses `unxt` [@unxt] for unit-aware
quantities and interoperates with `galax` [@galax] for potential evaluation.
This allows users to test flexible halo and disc models within a single compiled
pipeline.

Stream tracks are represented with splines rather than raw point sets.
`Potamides` provides utilities for constructing smooth ridge-line splines from
ordered stream points, optimizing knot placement, and computing the resulting
tangent vectors, principal normals, and scalar curvature. The package also
includes specific handling for locally straight segments where naive curvature
treatments might bias the likelihood.

Real stream systems are often fragmented or only partly observed. To address
this, `Potamides` evaluates segment-wise likelihoods and combines them across
multiple structures, treating curvature inference as a unified process over an
entire galaxy.

Performance is highly optimized. Starting from an annotated stream, `Potamides`
infers a ridge-line spline via gradient-based optimization in 10–30 seconds on a
2023 M2 MacBook Pro. Posterior refinement with NUTS [@Hoffman:2014] typically
requires another 20 seconds. The potential-inference stage evaluates the
curvature likelihood on a dense grid of $10^6$ parameter points—marginalizing
over roughly 50 spline realizations—in about 15 seconds per stream segment. For
a galaxy with three segments, an end-to-end analysis runs in under 15 minutes on
standard laptop hardware. On a single GPU, the combined spline-posterior and
potential-evaluation stages execute in roughly one minute. This speed supports
the rapid evaluation of many potential families, making it easier to test model
dependencies without committing early to a single dynamical description.

# Research impact statement

`Potamides` is actively used in current research workflows. Specifically, it is
being used in a Euclid Key Paper analysis of stellar streams in the Q1 data
release and in a separate study of streams in the Stream Legacy Survey. Both
projects rely on the package as the primary implementation for curvature-based
inference.

The software demonstrates immediate scientific value by providing tested,
high-performance, reproducible capabilities. Notably, `Potamides` successfully
reproduces the foundational research results of [@Nibauer:2023]. By streamlining
the process from annotating stream segments to calculating potential likelihoods
on standard hardware, `Potamides` serves as a highly practical tool for
researchers exploring gravitational potentials through stream morphology.

# AI usage disclosure

Generative AI tools were used during development to assist with refactoring,
drafting, and documentation. They were also used for copy-editing during
manuscript preparation. All AI-assisted outputs were reviewed and validated by
the human authors, who made all scientific, methodological, and software-design
decisions.

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
