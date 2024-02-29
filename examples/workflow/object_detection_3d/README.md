# Example Integration: Object Detection (3D)

This example integration uses the image and LIDAR data from the KITTI dataset to
demonstrate how to test 3D Object Detection on Kolena.

## Setup

This project uses [Poetry](https://python-poetry.org/) for packaging and Python dependency management. To get started,
install project dependencies from [`pyproject.toml`](./pyproject.toml) by running:

```shell
poetry update && poetry install
```

## Usage

This example integration will use pre-trained models provided by
[MMDetection3D](https://github.com/open-mmlab/mmdetection3d/blob/main/docs/en/model_zoo.md) library.
Please follow the [Installation](https://mmdetection3d.readthedocs.io/en/latest/get_started.html#installation)
and set up your environment.

### Data preparation

We will use the same data directory structure as detailed in
[KITTI dataset preparation guide](https://mmdetection3d.readthedocs.io/en/latest/advanced_guides/datasets/kitti.html),
replacing the root directory in the guide with current project directory.

After downloading the dataset, we provide a script to process them into a format that suitable for integration with
Kolena platform.

```
$ poetry run python3 object_detection_3d/prepare_test_samples.py --help
usage: prepare_test_samples.py [-h] datadir remote-prefix output

positional arguments:
  datadir        KITTI dataset dir
  remote_prefix  Prefix of cloud storage of KITTI raw data
  output         output file

options:
  -h, --help     show this help message and exit
$ poetry run python3 object_detection_3d/prepare_test_samples.py data/kitti s3://mybucket/kitti/3d-object-detection \
    test_samples_data.json
```

The script generates a file containing structured test samples and ground truths. This file will be used in
[Kolena integration](#kolena-integration) to create a test suite. The script also creates pointcloud files for each raw
velodyne binary file under `$datadir/velodyne_pcd`. Upload the camera images and pointcloud files to the cloud storage
location specified in the argument `remote_prefix`, mirroring the directory structure as described in
[Data preparation](#data-preparation) section, you can then view the images and visualize pointcloud data in
[Kolena studio](https://app.kolena.com/redirect/studio) after a test suite is created in
[Kolena Integration](#kolena-integration).

### Test the model

In this example, we will use [PointPillars](https://github.com/open-mmlab/mmdetection3d/tree/main/configs/pointpillars)
baseline. You can choose other suitable models with the same process.

```
mim download mmdet3d --config pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class --dest checkpoints
```

We provide a script to test the model on KITTI dataset that is easy to integrate with Kolena.

```
$ poetry run python3 object_detection_3d/prepare_inference_results.py --help
usage: prepare_inference_results.py [-h] [--device DEVICE] [--result-file RESULT_FILE] datadir config checkpoint

positional arguments:
  datadir               KITTI dataset dir
  config                Config file
  checkpoint            Checkpoint file

options:
  -h, --help            show this help message and exit
  --device DEVICE       Device used for inference
  --result-file RESULT_FILE
                        Result file
$ poetry run python3 object_detection_3d/prepare_inference_results.py --device cpu \
    data/kitti checkpoints/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class.py \
    checkpoints/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20220301_150306-37dc2420.pth
```

The model inference results are stored in `result-file`, which can be passed into seeding script, explained in the
next section.

### Kolena integration

First, ensure that the `KOLENA_TOKEN` environment variable is populated in your environment. See our
[initialization documentation](https://docs.kolena.com/installing-kolena/#initialization) for details.

This project defines two scripts that perform the following operations:

1. [`seed_test_suite.py`](object_detection_3d/seed_test_suite.py)
   creates a test suite `KITTI 3D Object Detection :: training :: metrics` using KITTI dataset for 3D object detection.

2. [`seed_test_run.py`](object_detection_3d/seed_test_run.py)
   tests a specified model, e.g. `second_hv_secfpn_8xb6-80e_kitti-3d-3class`, on the above test suites.

Command line arguments are defined within each script to specify what model to use and what test suite to seed/evaluate.
Run a script using the `--help` flag for more information:

```
$ poetry run python3 object_detection_3d/seed_test_suite.py --help
usage: seed_test_suite.py [-h] [--test-suite TEST_SUITE] sample_file

positional arguments:
  sample_file           File containing test sample and ground truth data

options:
  -h, --help            show this help message and exit
  --test-suite TEST_SUITE
                        Optionally specify a name for the created test suite.

$ poetry run python3 object_detection_3d/seed_test_suite.py test_samples_data.json
```

```
$ poetry run python3 object_detection_3d/seed_test_run.py --help
usage: seed_test_run.py [-h] [--test-suite TEST_SUITE] model model_results_file

positional arguments:
  model                 Model name.
  model_results_file    Name of model results file.

options:
  -h, --help            show this help message and exit
  --test-suite TEST_SUITE
                        Name of test suite to test.
$ poetry run python3 object_detection_3d/seed_test_run.py pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class \
    results.json
```
