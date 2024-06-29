---
description: How to find common model regressions and failures
---

# Difficulty Score

Difficulty scores are automatically computed within Kolena to surface datapoints that commonly contribute to poor
model performance. Difficulty scores consider a user's custom
[Quality Standard](../dataset/core-concepts/index.md#quality-standard) configuration to make
an informed assessment of which datapoints lead to the greatest recurring problems across all models using multiple
performance indicators. Difficulty scores range from 0 to 1, where a lower difficulty score
indicates that models produce the ideal datapoint-level metrics (e.g. lower inference time, higher accuracy),
and a higher difficulty score indicates that models consistently face problems or "difficulty"
(e.g. longer inference time, lower BLEU scores, and/or lower recall).

When one model is selected in Studio, the difficulty score of a datapoint is the difficulty score derived
from that model's results. When more than one model is considered, the overall difficulty score of a datapoint is
the average value from each difficulty score from each model's results.

??? "Using Difficulty Scores for Regression Testing"
    With two models called `old` and `new` are selected in Studio so users see two model-level difficulty scores,
    and one overall difficulty score for any datapoint. With a filter for
    `new.difficulty_score > old.difficulty_score`, we find all the datapoints that performed worse in the
    `new` model, which highlights the regressions.

    With a filter for `datapoint.difficulty_score > 0.9`, we see all the datapoints that significantly struggle
    across both the `old` and `new` models, which are common failures that persist over different model iterations.

## Implementation Details

Suppose we have defined quality standards composed of various metrics and performance indicators. Some of these will
be metrics like `ROUGE` or `accuracy` where higher values are better (`HIB`, higher is better), but it is
desirable for `word_error_rate` or `cost` to be minimized (`LIB`, lower is better).

Below is an example of how difficulty scores are computed for a single model using
`LIB(cost)`, `HIB(recall)`, and `HIB(accuracy)`:

```py
import pandas as pd

# names of features where "lower is better"
LIB = ['cost']

# names of features where "higher is better"
HIB = ['recall', 'accuracy']

# the name of the column for unique identifiers to a datapoint
id_column = 'id'


quality_standards = LIB + HIB
weights = [1/len(quality_standards)] * len(quality_standards) # weighting can be customized


model_results_csv = "path-to-first-model-results.csv"
df = pd.read_csv(model_results_csv, usecols=[id_column] + quality_standards)


def add_model_difficulty_score(df, lib, hib, weights):
    """
    Adds difficulty scores to datapoints in dataframe provided a quality standard configuration.
    """

    qs = lib + hib
    for col in qs:
        df[col] = df[col].astype(float)

        # invert HIB values such that higher values yield lower difficulty scores
        if col in hib:
            df[col] = df[col] * -1.0

        # normalize the column
        min_val = df[col].min()
        max_val = df[col].max()
        df[col] = (df[col] - min_val) / (max_val - min_val)

    df["difficulty_score"] = df[qs].dot(weights) # weighted sum
    return df


df = add_model_difficulty_score(df, LIB, HIB, weights)
```

## Example

1. Begin with a CSV of model results.

    | id | recall (HIB) | cost (LIB) | accuracy (HIB) |
    | --- | --- | --- | --- |
    | `1` | `0.10` | `3.14` | `0.50` |
    | `2` | `0.50` | `0.90` | `0.80` |
    | `3` | `0.90` | `0.01` | `0.99` |
    | `4` | `0.60` | `0.50` | `0.55` |

2. Invert every `HIB` column.

    | id | recall | cost | accuracy |
    | --- | --- | --- | --- |
    | `1` | `-0.10` | `3.14` | `-0.50` |
    | `2` | `-0.50` | `0.90` | `-0.80` |
    | `3` | `-0.90` | `0.01` | `-0.99` |
    | `4` | `-0.60` | `0.50` | `-0.55` |

3. Normalize every column.

    | id | recall | cost | accuracy |
    | --- | --- | --- | --- |
    | `1` | `1.00` | `1.00` | `1.00` |
    | `2` | `0.50` | `0.284` | `0.387` |
    | `3` | `0.00` | `0.00` | `0.00` |
    | `4` | `0.375` | `0.156` | `0.898` |

4. Compute difficulty scores using weighted sums for each datapoint.

    | id | recall | cost | accuracy | difficulty_score |
    | --- | --- | --- | --- | --- |
    | `1` | `1.00` | `1.00` | `1.00` | `1.00` |
    | `2` | `0.50` | `0.284` | `0.387` | `0.390` |
    | `3` | `0.00` | `0.00` | `0.00` | `0.00` |
    | `4` | `0.375` | `0.156` | `0.898` | `0.476` |

    Below is the math behind `id=2`'s difficulty score assuming equal weighting:
    ```txt
    difficulty_score = recall(0.5), cost(0.284), accuracy(0.387)
    = 1/3 * 0.5 + 1/3 * 0.284 + 1/3 * 0.387
    = 0.390
    ```

We have computed a new column of difficulty scores for each datapoint based on the quality standards set by the user.
If we were to add a new model, then the overall difficulty score would be the average of difficulty scores
across each model.

| id | model_1_ds | model_2_ds | model_3_ds | difficulty_score (ds) |
| --- | --- | --- | --- | --- |
| `1` | `0.3` | `0.3` | `0.3` | `0.30` |
| `2` | `0.1` | **`0.9`** | `0.1` | **`0.37`** |
| `3` | `0.4` | **`0.2`** | `0.6` | **`0.4`** |

From the table above, we see that the 2nd datapoint performs very poorly on the 2nd model with a difficulty score of
`0.9`, while the 3rd datapoint has `0.2` just underneath. However, the computed `difficulty_score` values indicate
that the 3rd datapoint is repeatedly the most challenging datapoint for the models based on the defined
quality standard.

### Difficulty Scores for Task Metrics

Difficulty scores for task metrics are aggregate metrics that do not offer datapoint-level details
of performance. However, the information provided to an aggregate metric is sufficient in establishing
difficulty scores at the datapoint level, similar to datapoint-level `inference_time` or `cost`.

#### Binary Classification and Regression

The difficulty score is the absolute error between the model result and the ground truth value.

Suppose in a binary classification problem, a model's inference is binarized by the threshold of `0.5`,
so the positive class would be defined by values `0.5` to `1.0`, and values of the negative class would
be from `0.0` to `0.5`.

| id | ground_truth | model_result | delta | difficulty_contribution |
| --- | --- | --- | --- | --- |
| `1` | `1`| `0.01` | `0.99` | `1.00` |
| `2` | `1`| `0.49` | `0.51` | `0.51` |
| `3` | `1`| `0.50` | `0.50` | `0.50` |
| `4` | `1`| `0.80` | `0.20` | `0.19` |
| `5` | `0`| `0.01` | `0.01` | `0.00` |
| `6` | `0`| `0.49` | `0.49` | `0.49` |
| `7` | `0`| `0.50` | `0.50` | `0.50` |
| `8` | `0`| `0.80` | `0.80` | `0.81` |

In the case of regression problems, difficulty can be measured in a similar way using the magnitude of the
difference between the ground truth and the model result.

| id | ground_truth | model_result | delta | difficulty_contribution |
| --- | --- | --- | --- | --- |
| `1` | `1`| `1` | `0` | `0.00` |
| `2` | `2`| `1` | `1` | `0.08` |
| `3` | `3`| `2` | `1` | `0.08` |
| `4` | `4`| `3` | `1` | `0.08` |
| `5` | `5`| `5` | `0` | `0.00` |
| `6` | `6`| `8` | `2` | `0.15` |
| `7` | `7`| `13` | `6` | `0.46` |
| `8` | `8`| `21` | `13` | `1.00` |

The greater the distance the model result is from the ground truth, the greater the difficulty of that
datapoint. These normalized `delta` column, called a `difficulty_contribution` column, becomes another
column in step 4 of the example above which is parallel to `cost` or `recall`.

| id | recall | difficulty_contribution | ... | difficulty_score |
| --- | --- | --- | --- | --- |
| `1` | `1.0`| `0.2` | ... | ... |
| `2` | `0.5`| `0.9` | ... | ... |

Then, it can be involved in the computation of the overall `difficulty_score` for the datapoint.

#### Multiclass Classification

The `delta` column for a datapoint of a multiclass classification task is simply the number of times a model
makes a mistake in classifying the datapoint. Like the binary classification and regression task, the
`difficulty_contribution` column normalizes `delta` to be used in computing the overall
datapoint `difficulty_score`.

#### Object Detection

The `delta` column for an object detection task is the F1 score computed using the total number of
[TP / FP / FN counts](./tp-fp-fn-tn.md). If recall is more important, this can become the default signal
instead of F1 scores at the datapoint level.
Like the other tasks, the `difficulty_contribution` column
normalizes `delta` to be used in computing the overall datapoint `difficulty_score`.

## Limitations and Biases

Difficulty scores require well-configured quality standards and perform better with multiple metrics and
indicators involved. Difficulty scores are not as useful unless multiple models are involved.

It is hard to interpret the value of a difficulty score directly, as it is an aggregate
signal reflecting quality standards relative to all models of interest with all their results.
