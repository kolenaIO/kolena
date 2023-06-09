# Using Kolena's Python Client

Testing on Kolena is conducted using the `kolena` Python package. You use the client to create and run tests from
your infrastructure that can be explored in our web platform.

`kolena` is released under the open-source [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0)
license. The package is [hosted on PyPI](https://pypi.org/project/kolena/) and can be installed using your
preferred Python package manager.

## Installation

The first step to start testing with Kolena is to install `kolena`. Client builds can be installed directly from
[PyPI](https://pypi.org/project/kolena/) using any Python package manager such as [pip](https://pypi.org/project/pip/):

```bash
pip install kolena
```

Or [Poetry](https://python-poetry.org/):

```bash
poetry add kolena
```

## Initialization

Once you have `kolena` installed, initialize a session with `kolena.initialize(token)`.

From [app.kolena.io/~/developer](https://app.kolena.io/redirect/developer), generate an API token and set the
`KOLENA_TOKEN` environment variable:

```bash
export KOLENA_TOKEN="********"
```

With the `KOLENA_TOKEN` environment variable set, initialize a client session:

```python
import os
import kolena

kolena.initialize(os.environ["KOLENA_TOKEN"], verbose=True)
```

By default, sessions have static scope and persist until the interpreter is exited.

Additional logging can be configured by specifying `initialize(..., verbose=True)`. All logging events are emitted as
Python standard library `logging` events from the `"kolena"` logger as well as to stdout/stderr directly.

<details>
<summary>blarg</summary>

::: kolena.initialize
</details>

## Supported Python Versions

`kolena` is compatible with all active Python versions.

| :fontawesome-brands-python: Python Version                        | Compatible `kolena` Versions |
| ----------------------------------------------------------------- | ---------------------------- |
| 3.11                                                              | ≥0.69                        |
| 3.10                                                              | _All Versions_               |
| 3.9                                                               | _All Versions_               |
| 3.8                                                               | _All Versions_               |
| 3.7                                                               | _All Versions_               |
| 3.6 (EOL: [December 2021](https://devguide.python.org/versions/)) | ≤0.46                        |
