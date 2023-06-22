<p align="center">
  <img src="https://docs.kolena.io/assets/images/wordmark-violet.svg" width="400" alt="Kolena" />
</p>

<p align='center'>
  <a href="https://pypi.python.org/pypi/kolena"><img src="https://img.shields.io/pypi/v/kolena" /></a>
  <a href="https://www.apache.org/licenses/LICENSE-2.0"><img src="https://img.shields.io/pypi/l/kolena" /></a>
  <a href="https://github.com/kolenaIO/kolena/actions"><img src="https://img.shields.io/github/checks-status/kolenaIO/kolena/trunk" /></a>
  <a href="https://codecov.io/gh/kolenaIO/kolena" ><img src="https://codecov.io/gh/kolenaIO/kolena/branch/trunk/graph/badge.svg?token=8WOY5I8SF1"/></a>
  <a href="https://docs.kolena.io"><img src="https://img.shields.io/badge/resource-docs-6434c1" /></a>
</p>

---

[Kolena](https://www.kolena.io) is a comprehensive machine learning testing and debugging platform to surface hidden
model behaviors and take the mystery out of model development. Kolena helps you:

- Perform high-resolution model evaluation
- Understand and track behavioral improvements and regressions
- Meaningfully communicate model capabilities
- Automate model testing and deployment workflows

This `kolena` package contains the Python client library for programmatic interaction with the Kolena ML testing
platform.

## Setup

Client builds can be installed directly from PyPI using any Python package manager such as pip:

```zsh
pip install kolena
```

Advanced use cases (eg. metrics computation) may require extra dependencies which can be installed by running:
```zsh
pip install kolena[metrics]
```

<details>
<summary>Installing with <a href="https://python-poetry.org/">Poetry</a></summary>
<br>
Install project dependencies by running

```zsh
poetry update && poetry install
```

Extra dependencies such as [Scikit-learn](https://scikit-learn.org/stable/) can be included by running
```zsh
poetry install --all-extras
```
</details>

For more information, see the [installation documentation](https://docs.kolena.io/testing-with-kolena/using-kolena-client#installation).

## Documentation

Visit [docs.kolena.io](https://docs.kolena.io/) for tutorial and usage documentation and the
[API Reference](https://app.kolena.io/api/developer/docs/html/index.html) for detailed `kolena` typing and
function documentation.
