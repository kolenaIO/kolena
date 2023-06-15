# Contributing to `kolena`

## Documentation

The documentation for `kolena`, hosted at [docs.kolena.io](https://docs.kolena.io/), is built out of this repo using
[MkDocs](https://www.mkdocs.org/).

To run the documentation server locally, run:

```
poetry run mkdocs serve -a localhost:8999
```

To build static documentation:

```
poetry run mkdocs build
```

Kolena sponsors the [mkdocs-material](https://squidfunk.github.io/mkdocs-material/) and
[mkdocstrings](https://mkdocstrings.github.io/) projects and uses "insiders" features from these projects. In order to
preserve the capability to build documentation without access to these private "insider" repos, the docs dependencies
declared in [`pyproject.toml`](pyproject.toml) reference the publicly available package sources.

To build the documentation with "insider" features, run the [`setup_insiders.sh`](docs/setup_insiders.sh) script:

```
./docs/setup_insiders.sh
```

Note that this script requires an SSH key in your environment with access to the [kolenaIO](https://github.com/kolenaIO)
organization.

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
