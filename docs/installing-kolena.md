---
icon: kolena/developer-16
---

# :kolena-developer-20: Installing `kolena`

Testing on Kolena is conducted using the [`kolena`](https://github.com/kolenaIO/kolena) Python package. You use the
client to create and run tests from your infrastructure that can be explored in our web platform.

`kolena` is released under the open-source [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0)
license. The package is [hosted on PyPI](https://pypi.org/project/kolena/) and can be installed using your
preferred Python package manager.

## Installation

The first step to start testing with Kolena is to install `kolena`. Client builds can be installed directly from
[PyPI](https://pypi.org/project/kolena/) using any Python package manager such as [pip](https://pypi.org/project/pip/)
or [Poetry](https://python-poetry.org/):

=== "`pip`"

    ```shell
    pip install kolena
    ```

=== "`poetry`"

    ```shell
    poetry add kolena
    ```

#### Extra Dependency Groups

Certain metrics computation functionality depends on additional packages like
[scikit-learn](https://scikit-learn.org/stable/). These extra dependencies can be installed via the `metrics` group:

=== "`pip`"

    ```shell
    pip install 'kolena[metrics]'
    ```

=== "`poetry`"

    ```shell
    poetry add 'kolena[metrics]'
    ```

## Initialization

Once you have `kolena` installed, sessions are automatically authenticated using any token present in the `KOLENA_TOKEN`
environment variable or in your `.netrc` file.

From the [:kolena-developer-16: Developer](https://app.kolena.com/redirect/developer) page, generate an API token and set
the `KOLENA_TOKEN` environment variable:

```bash
export KOLENA_TOKEN="********"
```

Alternatively, manually initialize a session by running the following:

```python
import kolena

kolena.initialize(verbose=True)
```

By default, sessions have static scope and persist until the interpreter is exited.
Additional logging can be configured by specifying `verbose=True`. See the documentation on
[`kolena.initialize`](./reference/initialize.md) for details.

!!! tip "Tip: `logging`"

    Integrate `kolena` into your existing logging system by filtering for events from the `"kolena"` logger. All log
    messages are emitted as both Python standard library [`logging`][logging] events as well as stdout/stderr messages.

## Supported Python Versions

`kolena` is compatible with all active Python versions.

| :fontawesome-brands-python: Python Version                        | :kolena-logo: Compatible `kolena` Versions |
|-------------------------------------------------------------------|--------------------------------------------|
| 3.12                                                              | ≥1.3                                       |
| 3.11                                                              | ≥0.69                                      |
| 3.10                                                              | _All Versions_                             |
| 3.9                                                               | _All Versions_                             |
| 3.8                                                               | _All Versions_                             |
| 3.7 (EOL: [June 2023](https://devguide.python.org/versions/))     | ≤0.99                                      |
| 3.6 (EOL: [December 2021](https://devguide.python.org/versions/)) | ≤0.46                                      |
