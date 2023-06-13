---
icon: kolena/classification-16
search:
  exclude: true
---

# :kolena-classification-20: `kolena.classification`

!!! warning "Legacy Warning"

    The `kolena.classification` module is considered **legacy** and should not be used for new projects.

    Please see `kolena.workflow` for customizable and extensible definitions to use for all new projects.

!["Dog" classification example from the Dogs vs. Cats dataset.](../../../assets/images/classification-dog.jpg)

`kolena.classification` supports the following types of classification models:

| Classification Type | Description                                                                                                                                     |
| ------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| **Binary**          | Classification model predicts a single class, using a threshold on prediction confidence to bisect the test set                                 |
| **Multi-label**     | Classification model predicts multiple classes, with each prediction over a threshold considered positive (i.e. ensemble of binary classifiers) |

## Quick Links

- [`kolena.classification.TestImage`][kolena.classification.TestImage]: create images for testing
- [`kolena.classification.TestCase`][kolena.classification.TestCase]: create and manage test cases
- [`kolena.classification.TestSuite`][kolena.classification.TestSuite]: create and manage test suites
- [`kolena.classification.TestRun`][kolena.classification.TestRun]: test models on test suites
- [`kolena.classification.Model`][kolena.classification.Model]: create models for testing

::: kolena.classification
    options:
      members_order: alphabetical

## Metadata

::: kolena.classification.metadata
    options:
      members_order: alphabetical
