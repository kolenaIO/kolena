---
icon: octicons/package-24
---

# :octicons-package-24: `kolena.workflow`

`kolena.workflow` contains the definitions to build a workflow:

1. Design data types, including any [`annotations`](annotation) or [`assets`](asset):

    - [`TestSample`][kolena.workflow.TestSample]: model inputs, e.g. images, videos, documents
    - [`GroundTruth`][kolena.workflow.GroundTruth]: expected model outputs
    - [`Inference`][kolena.workflow.Inference]: real model outputs

2. Define metrics and how they are computed:

    - [`Evaluator`][kolena.workflow.Evaluator]: metrics computation engine

3. Create tests:

    - [`TestCase`][kolena.workflow.TestCase]: a test dataset, or a slice thereof
    - [`TestSuite`][kolena.workflow.TestSuite]: a collection of test cases

4. Test models:

    - [`Model`][kolena.workflow.Model]: descriptor for a model
    - [`test`][kolena.workflow.test]: interface to run tests
