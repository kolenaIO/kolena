# Example Integration: Automatic Speech Recognition
This example integration uses the [Whisper](https://github.com/openai/whisper) and [Wav2Vec2](https://huggingface.co/facebook/wav2vec2-base) models, and [LibriSpeech dataset](https://www.openslr.org/12) to demonstrate how to test speech recognition problems on Kolena.

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

### Automatic Speech Recognition on LibriSpeech

This project defines two scripts that perform the following operations:

1. [`seed_test_suite.py`](automatic_speech_recognition/seed_test_suite.py) creates the following test suite:

    - `LibriSpeech`, stratified by `audio duration`, `tempo`, and `speaker sex`.

    Run this command to seed the default test suite:
    ```shell
    poetry run python3 automatic_speech_recognition/seed_test_suite.py
    ```


2. [`seed_test_run.py`](automatic_speech_recognition/seed_test_run.py) tests a specified model, e.g. `whisper-1-translate`, on the above test suite.

    Run this command to evaluate the default models on the `LibriSpeech` test suite:
    ```shell
    poetry run python3 automatic_speech_recognition/seed_test_run.py
    ```

Command line arguments are defined within each script to specify what model to use and what test suite to seed/evaluate.
Run a script using the `--help` flag for more information:

```shell
$ poetry run python3 automatic_speech_recognition/seed_test_run.py --help
usage: seed_test_run.py [-h] [--models MODELS [MODELS ...]]
                        [--test-suites TEST_SUITES [TEST_SUITES ...]]

optional arguments:
  -h, --help            show this help message and exit
  --models MODELS [MODELS ...]
                        Name(s) of model(s) in directory to test
  --test-suites TEST_SUITES [TEST_SUITES ...]
                        Name(s) of test suite(s) to test.
```
