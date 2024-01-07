# Example Integration: Speaker Diarization

This example integration uses the Google Cloud Speech-To-Text model,
and [ICSI-Corpus dataset](https://groups.inf.ed.ac.uk/ami/icsi/) to demonstrate how to test speaker diarization problems
on Kolena.

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

### Speaker Diarization on ICSI-Corpus

This project defines two scripts that perform the following operations:

1. [`upload_dataset.py`](speaker_diarization/upload_dataset.py) creates a dataset named
   `ICSI-Corpus` by default.

   Run this command to create the default dataset:
    ```shell
    poetry run python3 speaker_diarization/upload_dataset.py
    ```

2. [`upload_results.py`](speaker_diarization/upload_results.py) uploads the results of model
   `gcp-stt-video` for the above dataset.

   Run this command to upload results of model `gcp-stt-video` for the above dataset:
    ```shell
    poetry run python3 speaker_diarization/upload_results.py
    ```

Command line arguments are defined within each script to specify the dataset name to create or
upload results for. Run a script using the `--help` flag for more information:

```shell
$ poetry run python3 speaker_diarization/upload_dataset.py --help
usage: upload_dataset.py [-h] [--dataset-name DATASET_NAME] [--sample-count SAMPLE_COUNT]

optional arguments:
  -h, --help            show this help message and exit
  --dataset-name DATASET_NAME
                        Name of the dataset
  --sample-count SAMPLE_COUNT
                        Number of samples to use, all samples are used if 0

$ poetry run python3 speaker_diarization/upload_results.py --help
usage: upload_results.py [-h] [--dataset-name DATASET_NAME] [--align-speakers] [--sample-count SAMPLE_COUNT]

optional arguments:
  -h, --help            show this help message and exit
  --dataset-name DATASET_NAME
                        Name of the dataset.
  --align-speakers      Specify whether to perform speaker alignment between the ground_truth and inference in the preprocessing step.
  --sample-count SAMPLE_COUNT
                        Number of samples to use, all samples are used if 0.
```

## Quality Standards Guide

Once the dataset and results have been uploaded to Kolena, visit [Kolena](https://app.kolena.io/redirect/) to
[explore the data and results](https://docs.kolena.io/dataset/quickstart/#step-3-explore-data-and-results).

Here are our [Quality Standards](https://docs.kolena.io/dataset/core-concepts/quality-standard/) recommendations for
speaker diarization:

### Metrics

1. mean(DetectionPrecision)
2. mean(DetectionRecall)
3. mean(DetectionAccuracy)
4. mean(DiarizationErrorRate)
5. mean(IdentificationErrorRate)

### Plots

1. Distribution: `datapoint.DetectionAccuracy`
2. Distribution: `datapoint.DiarizationErrorRate`
3. Distribution: `datapoint.IdentificationErrorRate`

### Test Cases

1. `datapoint."Average Amplitude"` (with `bin` = 5)
