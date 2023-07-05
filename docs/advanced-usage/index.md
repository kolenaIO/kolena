---
icon: kolena/rocket-16
hide:
  - toc
---

# :kolena-rocket-20: Advanced Usage

This section contains tutorial documentation for advanced features available in Kolena.

<div class="grid cards" markdown>
- [:octicons-container-24: Packaging for Automated Evaluation](./packaging-for-automated-evaluation.md)

    ---

    Package [metrics evaluation logic](../reference/workflow/evaluator.md) in a Docker container image to dynamically
    compute metrics on relevant subsets of your test data.
</div>

<div class="grid cards" markdown>
- [:kolena-diagram-tree-16: Nesting Test Case Metrics](./nesting-test-case-metrics.md)

    ---

    Report class-level metrics within a test case and test ensembles and pipelines of models by nesting aggregate
    metrics within your [`MetricsTestCase`][kolena.workflow.MetricsTestCase].
</div>
