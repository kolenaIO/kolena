# Example Integration: Face Recognition (1:1)

This example integration uses the [Labeled Faces in the Wild (LFW)](http://vis-www.cs.umass.edu/lfw/)
dataset and Face Recognition (FR) workflow to
demonstrate how to test and evaluate end-to-end FR (1:1) model pipelines on Kolena.
The evaluation stages are: face detection, keypoint extraction, and recognition. There are 5,749 unique identifications
from the complete dataset with 13,233 images. In this example, we are using 9,164 images with one genuine pair and one
imposter pair.

## Setup

This project uses [Poetry](https://python-poetry.org/) for packaging and Python dependency management. To get started,
install project dependencies from [`pyproject.toml`](./pyproject.toml) by running:

```shell
poetry update && poetry install
```

## Usage

The data for this example integration lives in the publicly accessible S3 bucket `s3://kolena-public-examples`.

First, ensure that the `KOLENA_TOKEN` environment variable is populated in your environment. See our
[initialization documentation](https://docs.kolena.com/installing-kolena/#initialization) for details.

This project defines two scripts that perform the following operations:

1. [`upload_dataset.py`](face_recognition_11/upload_dataset.py) uploads the LFW dataset.

    > **NOTE:**  Face bounding box and keypoint ground truths are inferred
    > from [RetinaFace](https://github.com/serengil/retinaface/) as they are not provided in the LFW dataset.
    > Also, for demo purposes, we have subsampled 18,328 pairs, made up of 9,164 unique images, from LFW.

2. [`upload_results.py`](face_recognition_11/upload_results.py) tests a FR model with different detector backend
    model on the LFW dataset.

Command line arguments are defined within each script to specify the dataset name to create or model to upload results
for. Run a script using the `--help` flag for more information:

```shell
$ poetry run python3 face_recognition_11/upload_dataset.py --help
usage: upload_dataset.py [-h] [--dataset DATASET]

optional arguments:
  -h, --help         show this help message and exit
  --dataset DATASET  Optionally specify a custom dataset name to upload.

$ poetry run python3 face_recognition_11/upload_results.py --help
usage: upload_results.py [-h] [--model {vgg-face,facenet512}] [--detector {mtcnn,dlib}] [--dataset DATASET]

optional arguments:
  -h, --help            show this help message and exit
  --model {vgg-face,facenet512}
                        Name of FR model to test.
  --detector {mtcnn,dlib}
                        Name of detector backend to test.
  --dataset DATASET     Optionally specify a custom dataset name to test.
```

## Quality Standards Guide

Once the dataset and results have been uploaded to Kolena, visit [Kolena](https://app.kolena.io/redirect/) to
[explore the data and results](https://docs.kolena.com/dataset/quickstart/#step-3-explore-data-and-results).

Here are our [Quality Standards](https://docs.kolena.com/dataset/core-concepts/quality-standard/) recommendations for
face recognition [1:1]:

### Metrics

1. F1 Score [detection]
2. mean(result.alignment_NRMSE) [alignment]
3. mean(result.recognition_FNMR) [recognition]

### Plots

1. `datapoint.race` vs. `mean(result.alignment_NRMSE)`
2. `datapoint.age` vs. `mean(result.recognition_FNMR)`
3. `datapoint.race` vs. `mean(result.recognition_FNMR)`
4. `datapoint.gender` vs. `mean(result.recognition_FNMR)`
5. Distribution of `result.recognition_genuine_similarity`
6. Distribution of `result.recognition_imposter_similarity`

### Test Cases

1. `datapoint.bbox.aspect_ratio`
2. `datapoint.gender`
3. `datapoint.race`
