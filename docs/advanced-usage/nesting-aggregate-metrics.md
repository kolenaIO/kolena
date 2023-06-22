---
icon: kolena/property-16
---

# :kolena-property-20: Reporting Class-Level Metrics within a Test Case

When computing [test case metrics][kolena.workflow.MetricsTestCase] in an
[evaluator](../reference/workflow/evaluator.md), in some cases it is desirable to compute multiple sets of aggregate
metrics within a given test case. Here are a few examples of scenarios where this is desirable:

- **Multiclass workflows**: for ML tasks with multiple classes, a given test case may contain samples from
    more than one class. While it's useful to report metrics aggregated across all classes using an
    [averaging method](../metrics/averaging-methods.md), it's also useful to see aggregate metrics computed for each of
    the classes.
- **Ensemble or pipeline models**: when testing an ensemble or pipeline of multiple models, it can be useful to
    see metrics from the output of the complete ensemble/pipeline  as well as metrics computed for each of the
    constituent models.

In these cases, it is possible to nest additional aggregate metrics records within a
[`MetricsTestCase`][kolena.workflow.MetricsTestCase] object returned from an evaluator.

## How to
