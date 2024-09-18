# Contributing to `kolena`

## Development environment setup

`kolena` uses [uv](https://github.com/astral-sh/uv) to manage dependencies. To set up this repository for development,
run:

```shell
uv venv && source .venv/bin/activate
uv pip install --all-extras -r pyproject.toml
uv tool run pre-commit install
```

### Install pre-commit checks

Pre-commit is used for static analysis and to ensure code style consistency. Please install it so warnings and
errors can be caught before committing code.

```
uv tool run pre-commit install
```

## Documentation

The documentation for `kolena`, hosted at [docs.kolena.com](https://docs.kolena.com/), is built out of this repo using
[MkDocs](https://www.mkdocs.org/).
Building the documentation locally requires installing the dependencies for `kolena` as well as additional
[OS-level dependencies for mkdocs-material](https://squidfunk.github.io/mkdocs-material/plugins/requirements/image-processing/#cairo-graphics).

To run the documentation server locally, run:

```
uv run mkdocs serve -a localhost:8999
```

To build static documentation:

```
uv run mkdocs build
```

Note that any OSError unable to find "cairo-2, "cairo", or "libcairo-2" can be solved with
`ln -s /opt/homebrew/lib/libcairo.2.dylib .`, which makes a link of cairo to your current directory.

Kolena sponsors the [mkdocs-material](https://squidfunk.github.io/mkdocs-material/) and
[mkdocstrings](https://mkdocstrings.github.io/) projects and uses "insiders" features from these projects. In order to
preserve the capability to build documentation without access to these private "insider" repos, the docs dependencies
declared in [`pyproject.toml`](pyproject.toml) reference the publicly available package sources.

To build the documentation with "insider" features, run the [`setup_insiders.sh`](docs/setup_insiders.sh) script:

```
./docs/setup_insiders.sh
```

Note that this script requires an SSH key in your environment with access to the [kolenaIO](https://github.com/kolenaIO)
organization. Additionally, this script modifies the `pyproject.toml` to reference the "insider" versions of the mkdocs
projects. Please take care to avoid merging these updates, as it will prevent contributors without insiders access from
running `kolena`.

After running `setup_insiders.sh`, add `--config-file mkdocs.insiders.yml` to the `serve` and `build` invocations above
to build documentation with all "insider" features enabled.

### Images

Images used in documentation are stored in [`docs/assets/images`](docs/assets/images) and tracked with
[Git LFS](https://git-lfs.com/). All images in this directory are automatically tracked as LFS assets. Ensure that you
have LFS installed in your environment and run the following command:

```shell
git lfs install
```

### Links

Use relative paths, e.g. `../path-to-file.md`, to link to other pages within the documentation. Relative links that
include the `.md` file extension are checked during build, and broken links will cause builds to fail. This verification
is not provided for absolute or external links. When linking to an index page, reference it explicitly, e.g.
`path/to/index.md`.
