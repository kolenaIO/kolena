# Example Integration: Speaker Diarization
This example integration uses the Google Cloud Speech-To-Text model, and [ICSI-Corpus dataset](https://groups.inf.ed.ac.uk/ami/icsi/) to demonstrate how to test speaker diarization problems on Kolena.

## Setup

This project uses [Poetry](https://python-poetry.org/) for packaging and Python dependency management. To get started,
install project dependencies from [`pyproject.toml`](./pyproject.toml) by running:

```shell
poetry update && poetry install
```

## Usage

The data for this example integration lives in the publicly accessible S3 bucket `s3://kolena-public-datasets`.

First, ensure that the `KOLENA_TOKEN` environment variable is populated in your environment. See our
[initialization documentation](https://docs.kolena.io/installing-kolena/#initialization) for details.

### Speaker Diarization on ICSI-Corpus

This project defines two scripts that perform the following operations:

1. [`seed_test_suite.py`](speaker_diarization/seed_test_suite.py) creates the following test suite:

    - `ICSI-Corpus`, stratified by `average-amplitude`.

    Run this command to seed the default test suite:
    ```shell
    poetry run python3 speaker_diarization/seed_test_suite.py
    ```


2. [`seed_test_run.py`](speaker_diarization/seed_test_run.py) tests a specified model, `gcp-stt-video`, on the above test suite.

    Run this command to evaluate the default model on the `ICSI-Corpus` test suite:
    ```shell
    poetry run python3 speaker_diarization/seed_test_run.py
    ```

Command line arguments are defined within each script to specify what model to use and what test suite to seed/evaluate.
Run a script using the `--help` flag for more information:

```shell
$ poetry run python3 speaker_diarization/seed_test_run.py --help
usage: seed_test_run.py [-h] [--align-speakers ALIGN_SPEAKERS]
options:
  -h, --help            show this help message and exit
  --align-speakers ALIGN_SPEAKERS
                        Specify whether to perform speaker alignment between the GT and Inf in the preprocessing step.
```
