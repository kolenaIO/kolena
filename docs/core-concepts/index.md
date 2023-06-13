---
icon: kolena/flag-16
hide:
  - toc
---

# :kolena-flag-20: Core Concepts

In this section, we'll get acquainted with the core concepts on Kolena, and learn in-depth about the various features
offered. For a brief introduction, see the [Quickstart Guide](../quickstart.md) or the
[Building a Workflow](../building-a-workflow.md) tutorial. For code-level API documentation, see the
[API Reference Documentation](../reference/workflow) for the `kolena` Python client.

<div class="grid cards" markdown>
- [:kolena-workflow-16: Workflow](workflow)

    ---

    Testing in Kolena is broken down by the type of ML problem you're solving, called a workflow. Any ML problem that
    can be tested can be modeled as a workflow in Kolena.
</div>

<div class="grid cards" markdown>
- [:kolena-test-suite-16: Test Cases & Test Suites](test-suites)

    ---

    Test cases and test suites are used to organize test data in Kolena.
</div>

<div class="grid cards" markdown>
- [:kolena-model-16: Test Cases & Test Suites](test-suites)

    ---

    In Kolena, a model is a deterministic transformation from [test samples](workflow.md#test-sample) to
    [inferences](workflow.md#inference).
</div>
