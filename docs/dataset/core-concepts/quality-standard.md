---
icon: kolena/quality-standard-16
search:
  boost: 2
---

# :kolena-quality-standard-20: Quality Standard


**Quality Standard** is the centralized interface where users interact with two fundamental components:

- **Test Coverage**: This encompasses a group of **Test Cases**, where each represents a division of the dataset by one of the fields of the datapoints.

- **Evaluation Criteria**: These are the metrics established for comparing model performances. You will be able to compare the results from different models based on these metrics.

## Configured Test Cases
**Configured Test Cases** is the way to divide dynamically your dataset into different test cases using one of the fields of the datapoints. For instance, if you have a dataset for face recognition, you can configure the test cases to divide this dataset by age, gender, race, etc.

## Evaluation Criteria
**Evaluation Criteria** represent dynamic metrics that can be defined from one of the fields of the datapoint results. These metrics can be calculated using a formula that depends on the type of fields. For instance, if the field is a number, you can define a formula like mean, median, min, max, standard deviation, or sum.

These metrics will enable you to compare different results from various models.


## Model Cards

**Model Cards** is the component of Kolena that allows you to compare the results of different models using the evaluation criteria previously defined.
