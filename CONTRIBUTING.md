# Contributing to potamides

Thank you for your interest in contributing! potamides is made for its users, so
contributions of all kinds are welcome and encouraged --- whether you are a
seasoned developer or have never contributed to an open-source project before.
You do not need to write code to help: reporting a bug, suggesting a feature,
improving the documentation, or sharing a scientific use case are all valuable.

By participating in this project, you agree to abide by our
[Code of Conduct](CODE_OF_CONDUCT.md), which helps keep this a welcoming and
inclusive community.

## Ways to contribute

Even if you have never used GitHub before, getting involved is straightforward.

### Report a bug or request a feature

Open an [issue](https://github.com/xggs-dev/potamides/issues). For bug reports,
please include enough detail for us to reproduce the problem:

- a minimal example that triggers the issue,
- the full error message or traceback,
- your operating system, Python version, and the versions of potamides and its
  key dependencies (`jax`, `numpy`, etc.).

### Contribute code or documentation

We welcome pull requests. If you are new to the GitHub workflow, GitHub's
[guide to contributing to projects](https://docs.github.com/en/get-started/exploring-projects-on-github/contributing-to-a-project)
is a good place to start, and we are happy to help you through the process — do
not hesitate to open a draft pull request or ask in Discussions.

For anything beyond a small fix, consider opening an issue first so we can
discuss the approach before you invest significant effort.

## Development setup

potamides uses [`uv`](https://docs.astral.sh/uv/) for environment and dependency
management and [`nox`](https://nox.thea.codes/) to run common tasks.

1. **Fork** the repository on GitHub, then **clone** your fork:

   ```bash
   git clone https://github.com/<your-username>/potamides.git
   cd potamides
   ```

2. **Create an environment and install** the package with its development
   dependencies:

   ```bash
   uv sync
   ```

   This installs potamides in editable mode along with the `dev` dependency
   group (tests, linters, docs, and nox).

3. **Install the pre-commit hooks** so formatting and linting run automatically
   on each commit:

   ```bash
   uv run pre-commit install
   ```

## Making changes

1. Create a branch for your work:

   ```bash
   git checkout -b my-feature
   ```

2. Make your changes, adding tests and documentation where appropriate. New code
   should be type-annotated — the package is checked with `mypy` in strict mode.

3. Run the checks (see below) and make sure they pass.

4. Commit your changes. This project follows the
   [Conventional Commits](https://www.conventionalcommits.org/) style with
   gitmoji (via [commitizen](https://commitizen-tools.github.io/commitizen/)).
   For example:

   ```text
   ✨ feat: add curvature-weighted likelihood
   🐛 fix: correct arc-length normalization
   📝 docs: clarify spline fitting example
   ```

5. Push your branch and open a
   [pull request](https://github.com/xggs-dev/potamides/pulls) against `main`.
   Describe what the change does and why; link any related issue.

## Running the checks

You can run everything through `nox`, which manages isolated environments for
you:

```bash
# Run the test suite
uv run nox -s pytest

# Run linters and type checks (pre-commit + pylint)
uv run nox -s lint

# Build the documentation
uv run nox -s docs
```

To list all available sessions:

```bash
uv run nox -l
```

If you prefer to run the tools directly within the synced environment:

```bash
uv run pytest
uv run pre-commit run --all-files
```

### A note on test data

Some tests compare numerical arrays and generated plots against stored baselines
(via `pytest-arraydiff` and `pytest-mpl`). If you intentionally change numerical
output or figures, regenerate the baselines with the corresponding nox sessions
(`make_test_arraydiff`, `make_test_mpl`) and review the diffs carefully before
committing.

## Continuous integration

When you open a pull request, the CI workflow runs the linters, type checks, and
test suite across supported Python versions. All checks must pass before a pull
request can be merged. If CI fails, click through to the logs to see what went
wrong — and feel free to ask for help if anything is unclear.

## Questions?

If you get stuck at any point, open an
[Issue](https://github.com/xggs-dev/potamides/issues) or ask in your pull
request.
