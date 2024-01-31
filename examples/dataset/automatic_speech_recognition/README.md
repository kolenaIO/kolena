# Example Integration: Automatic Speech Recognition

This example integration uses the [Whisper](https://github.com/openai/whisper)
and [Wav2Vec2](https://huggingface.co/facebook/wav2vec2-base) models,
and [LibriSpeech dataset](https://www.openslr.org/12)
to demonstrate how to test speech recognition problems on Kolena.

## Setup

This project uses [Poetry](https://python-poetry.org/) for packaging and Python dependency management. To get started,
install project dependencies from [`pyproject.toml`](./pyproject.toml) by running:

```shell
poetry update && poetry install
```

## Usage

The data for this example integration lives in the publicly accessible S3 bucket `s3://kolena-public-examples`.

First, ensure that the `KOLENA_TOKEN` environment variable is populated in your environment. See our
[initialization documentation](https://docs.kolena.io/installing-kolena/#initialization) for details.

### Automatic Speech Recognition on LibriSpeech

This project defines two scripts that perform the following operations:

1. [`upload_dataset.py`](automatic_speech_recognition/upload_dataset.py) creates the LibriSpeech dataset on Kolena

    Run this command to upload the dataset:

    ```shell
    poetry run python3 automatic_speech_recognition/upload_dataset.py
    ```

2. [`upload_results.py`](automatic_speech_recognition/upload_results.py) tests an ASR model, e.g. `whisper-default`.

    Run this command to evaluate the default model on the `LibriSpeech` dataset:

    ```shell
    poetry run python3 automatic_speech_recognition/upload_results.py
    ```

Command line arguments are defined within each script to specify what model to use.
Run a script using the `--help` flag for more information:

```shell
$ poetry run python3 automatic_speech_recognition/upload_results.py --help
usage: upload_results.py [-h] [--dataset DATASET] [{whisper-translate,whisper-default,wav2vec2}]

positional arguments:
  {whisper-translate,whisper-default,wav2vec2}
                        Name of the model to test.

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     Optionally specify a custom dataset name to test.
```

## Quality Standards Guide

Once the dataset and results have been uploaded to Kolena, visit [Kolena](https://app.kolena.io/redirect/) to
test the automatic speech recognition models. See our [QuickStart](https://docs.kolena.io/dataset/quickstart/) guide
for details.

Here are our [Quality Standards](https://docs.kolena.io/dataset/core-concepts/quality-standard/) recommendations for
this workflow:

### Metrics

1. [Word Error Rate](https://docs.kolena.io/metrics/wer-cer-mer)
2. [Character Error Rate](https://docs.kolena.io/metrics/wer-cer-mer/)
3. [Match Error Rate](https://docs.kolena.io/metrics/wer-cer-mer/)
4. Failure Rate (`rate(is_failure=true)`)

### Plots

1. `mean(result.character_error_rate)` vs. `datapoint.duration_seconds`
2. `mean(result.character_error_rate)` vs. `datapoint.tempo`

### Test Cases

1. `datapoint.speaker_sex`
2. `datapoint.duration_seconds`
