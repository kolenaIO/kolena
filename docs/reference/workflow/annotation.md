# Annotations: `kolena.workflow.annotation`

Annotations are visualized in Kolena as overlays on top of [`TestSample`][kolena.workflow.TestSample] objects.

The following annotation types are available:

- [`BoundingBox`][kolena.workflow.annotation.BoundingBox]
- [`Polygon`][kolena.workflow.annotation.Polygon]
- [`Polyline`][kolena.workflow.annotation.Polyline]
- [`Keypoints`][kolena.workflow.annotation.Keypoints]
- [`BoundingBox3D`][kolena.workflow.annotation.BoundingBox3D]
- [`SegmentationMask`][kolena.workflow.annotation.SegmentationMask]
- [`BitmapMask`][kolena.workflow.annotation.BitmapMask]
- [`ClassificationLabel`][kolena.workflow.annotation.ClassificationLabel]

For example, when viewing images in the Studio, any annotations (such as lists of
[`BoundingBox`][kolena.workflow.annotation.BoundingBox] objects) present in the
[`TestSample`][kolena.workflow.TestSample], [`GroundTruth`][kolena.workflow.GroundTruth],
[`Inference`][kolena.workflow.Inference], or [`MetricsTestSample`][kolena.workflow.MetricsTestSample] objects are
rendered on top of the image.

## Usage

The primary usage of annotations is in [`TestSample`][kolena.workflow.TestSample],
[`GroundTruth`][kolena.workflow.GroundTruth], and [`Inference`][kolena.workflow.Inference] types when
[building a workflow](../../building-a-workflow.md) and in [`MetricsTestSample`][kolena.workflow.MetricsTestSample]
when defining per-sample metrics and [implementing an evaluator](../../building-a-workflow.md#step-2-defining-metrics).

Annota TODO

### Extending Annotations

All annotation types can be extended with arbitrary fields.

```python
from dataclasses import dataclass
from typing import Optional

from kolena.workflow.annotation import LabeledBoundingBox

@dataclass(frozen=True)
class ExtendedLabeledBoundingBox(LabeledBoundingBox):
    # all fields from LabeledBoundingBox are inherited
    supercategory: str  # add additional fields as necessary
    subcategory: Optional[str] = None
```

This custom `ExtendedLabeledBoundingBox` can be used in [`TestSample`][kolena.workflow.TestSample],
[`GroundTruth`][kolena.workflow.GroundTruth], [`Inference`][kolena.workflow.Inference], and
[`MetricsTestSample`][kolena.workflow.MetricsTestSample] definitions like any other annotation. Any additional fields
(`supercategory` and `subcategory` in the above example) are available to visualize, sort, and filter in the
[:kolena-studio-16: Studio](https://app.kolena.io/redirect/studio).

## Reference

::: kolena.workflow.annotation
