---
icon: kolena/test-suite-16
status: new
---

# :kolena-test-suite-20: ThresholdedMetrics

!!! example "Experimental Feature"

    This pre-built workflow is an experimental feature. Experimental features are under active development and may
    occasionally undergo API-breaking changes.

Thresholded Metrics play a crucial role in specific contexts of model evaluation, particularly when the performance
of models is assessed based on varying threshold values. In such scenarios, these metrics are vital for accurately
interpreting the effectiveness of the models, as different thresholds can lead to markedly different performance
outcomes.

Here is an example of how `ThresholdedMetrics` can be used within a workflow:

```python
from kolena.workflow import MetricsTestSample
from kolena.workflow import ThresholdedMetrics


@dataclass(frozen=True)
class ClassThresholdedMetrics(ThresholdedMetrics):
    precision: float
    recall: float
    f1: float


@dataclass(frozen=True)
class TestSampleMetrics(MetricsTestSample):
    car: List[ClassThresholdedMetrics]
    pedestrian: List[ClassThresholdedMetrics]

# Creating an instance of metrics
metric = TestSampleMetrics(
    car=[
        ClassThresholdedMetrics(threshold=0.3, precision=0.5, recall=0.8, f1=0.615),
        # ...
    ],
    pedestrian=[
        ClassThresholdedMetrics(threshold=0.3, precision=0.6, recall=0.9, f1=0.72),
        # ...
    ],
)
```
