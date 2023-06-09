# Contributing to `kolena`

## Documentation

### Setup

The documentation for `kolena`, hosted at [docs.kolena.io](https://docs.kolena.io/), is built out of this repo using
[MkDocs](https://www.mkdocs.org/).

Kolena sponsors the [mkdocs-material](https://squidfunk.github.io/mkdocs-material/) and
[mkdocstrings](https://mkdocstrings.github.io/) projects and uses "insiders" features from these projects. To preserve
the capability to build documentation both with and without access to these private "insider" repos, this project
declares the `docs` and `docs-insiders` dependency groups that can be installed independently of one another.

Install the `docs` group to build documentation using publicly available `mkdocs-material` and `mkdocstrings` packages:

```
poetry update --with docs
```

Install the `docs-insiders` group to build documentation using private forks of the "insiders" packages:

```
poetry update --with docs-insiders
```

### Usage

To run the documentation server locally, run:

```
poetry run mkdocs serve -a localhost:8999
```

To build static documentation:

```
poetry run mkdocs build
```
