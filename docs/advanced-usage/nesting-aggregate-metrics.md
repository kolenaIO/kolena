---
icon: kolena/property-16
---

# :kolena-property-20: Nesting Aggregate Metrics

When computing [test case metrics][kolena.workflow.MetricsTestCase] in an
[evaluator](../reference/workflow/evaluator.md), in some cases it is desirable to compute multiple sets of aggregate
metrics within a given test case. Here are a few examples of scenarios where this pattern might be warranted:

- **Multiclass workflows**: for ML tasks with multiple classes, a given test case may contain samples from
    more than one class. While it's useful to report metrics aggregated across all classes using an
    [averaging method](../metrics/averaging-methods.md), it's also useful to see aggregate metrics computed for each of
    the classes.
- **Ensemble or pipeline models**: when testing an ensemble or a pipeline containing multiple models, it can be useful
    to see metrics from the output of the complete ensemble/pipeline  as well as metrics computed for each of the
    constituent models.

In these cases, it is possible to nest additional aggregate metrics records within a
[`MetricsTestCase`][kolena.workflow.MetricsTestCase] object returned from an evaluator. In this tutorial, we'll learn
how to use this API to report class-level or other nested aggregate metrics within a test case.

## Example: Multiclass Object Detection

Let's consider the case of a multiclass object detection task with objects of type `Airplane`, `Boat`, and `Car`.
When a test case contains images with each of these three classes, test-case-level metrics are the average (e.g.
[micro](../metrics/averaging-methods.md#micro-average), [macro](../metrics/averaging-methods.md#macro-average), or
[weighted](../metrics/averaging-methods.md#weighted-average)) of class-level metrics across each of these three classes.

Using [macro-averaged](../metrics/averaging-methods.md#macro-average) precision, recall, and F1 score, and mean average
precision score (mAP) across all images in the test case.

| Test Case | # Images | `Precision_macro` | `Recall_macro` | `F1_macro` | `mAP` |
| --- | --- | --- | --- | --- | --- |
| Example Scenario | 10 | 0.5 | 0.5 | 0.5 | 0.5 |

These metrics would be defined:

```python
from dataclasses import dataclass

from kolena.workflow import MetricsTestCase

@dataclass(frozen=True)
class AggregateMetrics(MetricsTestCase):
    # Test Case, # Images are automatically populated
    Precision_macro: float
    Recall_macro: float
    F1_macro: float
    mAP: float
```

These metrics tell us how well the model performs in `Example Scenario` across all classes, but they don't tell us
anything about per-class model performance. Within each test case, we'd also like to see precision, recall, F1, and AP
scores:

| `Class` | `N` | `Precision` | `Recall` | `F1` | `AP` |
| --- | --- | --- | --- | --- | --- |
| `Airplane` | 5 | 0.5 | 0.5 | 0.5 | 0.5 |
| `Boat` | 5 | 0.5 | 0.5 | 0.5 | 0.5 |
| `Car` | 5 | 0.5 | 0.5 | 0.5 | 0.5 |

We can report these class-level metrics alongside the macro-averaged overall metrics by nesting
[`MetricsTestCase`][kolena.workflow.MetricsTestCase] definitions:

```python
from dataclasses import dataclass
from typing import List

from kolena.workflow import MetricsTestCase

@dataclass(frozen=True)
class PerClassMetrics(MetricsTestCase):
    Class: str  # name of the class corresponding to this record
    N: int  # number of samples containing this class
    Precision: float
    Recall: float
    F1: float
    AP: float

@dataclass(frozen=True)
class AggregateMetrics(MetricsTestCase):
    # Test Case, # Images are automatically populated
    Precision_macro: float
    Recall_macro: float
    F1_macro: float
    mAP: float
    PerClass: List[PerClassMetrics]
```

Now we have the definitions to tell us everything we need to know about model performance within a test case:
`AggregateMetrics` describes overall performance across all classes within the test case, and `PerClassMetrics`
describes performance for each of the given classes within the test case.

### Naming a Record

...

### Statistical Significance

...

## Conclusion

In this tutorial, we learned how to use the [`MetricsTestCase`][kolena.workflow.MetricsTestCase] API to define
class-level metrics within a test case. Nesting aggregate metrics is desirable for multiclass workflows, as well as when
testing ensembles of models or testing model pipelines.
