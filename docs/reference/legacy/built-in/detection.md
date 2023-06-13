---
icon: kolena/detection-16
search:
  exclude: true
---

# :kolena-detection-20: `kolena.detection`

!!! warning "Legacy Warning"

    The `kolena.detection` module is considered **legacy** and should not be used for new projects.

    Please see `kolena.workflow` for customizable and extensible definitions to use for all new projects.

![Object detection example from the Common Objects in Context (COCO) dataset.](../../../assets/images/detection-airplane.jpg)

Object detection models attempt to localize and classify objects in an image. Kolena supports single-class and
multi-class object detection models identifying objects with rectangular (object detection) and arbitrary (instance
segmentation) geometry.

!!! note
    **Instance Segmentation** and **Object Detection** are functionally equivalent, differing only in the geometry of the
    detected object. For brevity, this documentation discusses object detection only.

## Quick Links

- [`kolena.detection.TestImage`][kolena.detection.TestImage]: create images for testing
- [`kolena.detection.TestCase`][kolena.detection.TestCase]: create and manage test cases
- [`kolena.detection.TestSuite`][kolena.detection.TestSuite]: create and manage test suites
- [`kolena.detection.TestRun`][kolena.detection.TestRun]: test models on test suites
- [`kolena.detection.Model`][kolena.detection.Model]: create models for testing

::: kolena.detection
    options:
      members_order: alphabetical

## Ground Truth

::: kolena.detection.ground_truth
    options:
      members_order: alphabetical

## Inference

::: kolena.detection.inference
    options:
      members_order: alphabetical

## Metadata

::: kolena.detection.metadata
    options:
      members_order: alphabetical
