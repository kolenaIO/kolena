# Example Integration: 2D Object Detection with Extended GroundTruth

This example integration is a copy of [`object_detection_2d` example](../object_detection_2d/) with
an additional field added to `GroundTruth`, demonstrating how you can extend a pre-built workflow definition.

## Setup

This project uses [Poetry](https://python-poetry.org/) for packaging and Python dependency management. To get started,
install project dependencies from [`pyproject.toml`](./pyproject.toml) by running:

```shell
poetry update && poetry install
```

## Usage

All data for this example integration lives in the publicly accessible S3 bucket `s3://kolena-public-datasets`.

First, ensure that the `KOLENA_TOKEN` environment variable is populated in your environment. See our
[initialization documentation](https://docs.kolena.io/installing-kolena/#initialization) for details.

This project defines two scripts that perform the following operations:

1. [`seed_test_suite.py`](object_detection_2d_extend/seed_test_suite.py) creates the following test suite:

    - `coco-2014-val :: transportation by brightness [Object Detection Extended]`, stratified by `light`, `normal`, and `dark`
        brightness
    - `coco-2014-val :: transportation by bounding box size [Object Detection Extended]`, stratified by bounding box size:
        `small`, `medium` and `large`.


2. [`seed_test_run.py`](object_detection_2d_extend/seed_test_run.py) tests the following models on the above test suites:
  `yolo_r`, `yolo_x`, `mask_cnn`, `faster_rcnn`, `yolo_v4s`, and `yolo_v3`. Information about these models can be
  found in [`constants.py`](object_detection_2d_extend/constants.py).

Command line arguments are defined within each script to specify what model to use and what test suite to
  seed/evaluate. Run a script using the `--help` flag for more information:

```shell
$ poetry run python3 object_detection_2d_extend/seed_test_run.py --help
usage: seed_test_run.py [-h] [--test_suite TEST_SUITE] {yolo_r,yolo_x,mask_cnn,faster_rcnn,yolo_v4s,yolo_v3}

positional arguments:
  {yolo_r,yolo_x,mask_cnn,faster_rcnn,yolo_v4s,yolo_v3}
                        The alias of the model to test.

optional arguments:
  -h, --help            show this help message and exit
  --test-suite TEST_SUITE
                        Optionally specify a test suite to test. Test against all available test suites when unspecified.
```

## Pre-built Workflow Extension

A pre-built workflow can be extended simply by re-defining the [`workflow`](object_detection_2d_extend/workflow.py). In
this example, we added a new boolean field `occluded` to each `GroundTruth` `LabeledBoundingBox` by extending it.

```python
@dataclass(frozen=True)
class ExtendedBoundingBox(LabeledBoundingBox):
    occluded: bool

@dataclass(frozen=True)
class GroundTruth(BaseGroundTruth):
    bboxes: List[ExtendedBoundingBox]
```

And define a new workflow with the extended `GroundTruth` class:

```python
_workflow, TestCase, TestSuite, Model = define_workflow(
    "Object Detection Extended",
    TestSample,
    GroundTruth,
    Inference,
)
```

Note that `GroundTruth` and `ExtendedBoundingBox` are now imported from `workflow.py` instead.
