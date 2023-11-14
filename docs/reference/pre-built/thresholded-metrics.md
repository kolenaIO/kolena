---
icon: kolena/metrics-16
status: new
---

# :kolena-metrics-20: Thresholded Metrics

!!! tip "Key Component"

    `ThresholdedMetrics` is a fundamental class in the Kolena workflow for handling metrics tied to specific thresholds. It ensures data integrity and consistency across various metrics evaluations.

Thresholded Metrics are an essential aspect of evaluating and comparing the performance of models, especially in scenarios where different threshold values can significantly impact the interpretation of the model's effectiveness.

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
