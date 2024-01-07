---
icon: kolena/quality-standard-16
search:
  boost: 2
---

# :kolena-quality-standard-20: Quality Standard

A **Quality Standard** tracks a standardized process for how a team evaluates a model's performance on a dataset.
Users may define and manage quality standards for a dataset in the Kolena web application from a that dataset's
`Quality Standards` tab. Once defined, a quality standard provides a well-defined framework for easily understanding and
comparing future model results.

A Quality Standard is composed of [Test Cases](#test-cases) and [Metrics](#metrics).

## Test Cases

Test cases allow users to evaluate their datasets at various levels of division, providing visibility into how models
perform at differing subsets of the full dataset, and mitigating failures caused by
[hidden stratifications](https://www.kolena.com/blog/best-practices-for-ml-model-testing).

Kolena supports easy test case creation through dividing a dataset along categorical or numeric datapoint properties.
For example, if you have a dataset with images of faces of individuals, you may wish to create a set of test cases that
divides your dataset by `datapoint.race` (categorical) or `datapoint.age` (numeric).

The datasets quickstart provides a more hands-on example of
[defining test cases](../quickstart.md/#define-test-cases).

## Metrics

Metrics describe the criteria used to evaluate the performance of a model and compare it with other models over a given
dataset and its test cases.

Kolena supports defining metrics by applying standard aggregations over datapoint level results or by leveraging
common machine learning aggregations, such as [Precision](../../../metrics/precision) or
[F1 Score](../../../metrics/f1-score). Once defined, users may also specify highlighting for metrics, indicating if
`Higher is better`, or if `Lower is better`.

The datasets quickstart provides a more hands-on example of
[defining metrics](../quickstart.md/#define-metrics).

## Model Comparison

Once you've defined your test cases and metrics, you can view and compare model results in the `Quality Standards` tab,
which provides a quick and standardized high level overview of which models perform best over your different test cases.

For step-by-step instructions, take a look at the quickstart for
[model comparison](../quickstart.md/#step-5-compare-models).

## Debugging

The `Debugger` tab of a dataset allows users to experiment with test cases and metrics without saving them off to the
team level quality standards. This allows users to search for meaningful test cases and experiment with different
metrics with the confidence that they can safely save these updated values to their quality standards when comfortable,
without the risk of accidentally replacing what the team has previously defined. This also provides a view for
visualizing results and relations in plots.

For step-by-step instructions, take a look at the quickstart for
[results exploration](../quickstart.md/#step-3-explore-data-and-results).
