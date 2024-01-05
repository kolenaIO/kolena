---
icon: kolena/area-of-interest-16
hide:
  - toc
---

!!! warning "Legacy Warning"

    Content in this section reflects **outdated practices** or **deprecated features**. It's recommended to avoid using these in new developments.

    While existing implementations using these features will continue to receive support, we strongly advise adopting the latest standards and tools for new projects to ensure optimal performance and compatibility. For more information and up-to-date practices, please refer to our newest documentation at [docs.kolena.io](https://docs.kolena.io).


# :kolena-area-of-interest-20: Core Concepts

In this section, we'll get acquainted with the core concepts on Kolena, and learn in-depth about the various features
offered. For a brief introduction, see the [Quickstart Guide](../quickstart.md) or the
[Building a Workflow](../building-a-workflow.md) tutorial. For code-level API documentation, see the
[API Reference Documentation](../reference/index.md) for the `kolena` Python client.

<div class="grid cards" markdown>
- [:kolena-workflow-16: Workflow](workflow.md)

    ---

    Testing in Kolena is broken down by the type of ML problem you're solving, called a workflow. Any ML problem that
    can be tested can be modeled as a workflow in Kolena.
</div>

<div class="grid cards" markdown>
- [:kolena-test-suite-16: Test Cases & Test Suites](test-suite.md)

    ---

    Test cases and test suites are used to organize test data in Kolena.
</div>

<div class="grid cards" markdown>
- [:kolena-model-16: Models](model.md)

    ---

    In Kolena, a model is a deterministic transformation from [test samples](workflow.md#test-sample) to
    [inferences](workflow.md#inference).
</div>
