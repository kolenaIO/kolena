---
icon: kolena/diagram-tree-16

---

# :kolena-diagram-tree-20: Thresholded Results

## When to Use Thresholded Results

- Comparing Metrics at Different Thresholds.
- Calculating the [Precision-Recall (PR) Curve](../../metrics/pr-curve.md) dynamically based on stratification.
- Computing [Average Precision](../../metrics/average-precision.md) dynamically based on stratification.

## Format and upload thresholded results

A thresholded object is a list of dictionaries. Each dictionary contains:

- A required numeric field `threshold` (number).
- An optional string field `label` (string).
- Any other metrics for the label at that specific threshold (number).

For instance, in a semantic segmentation problem:

```py
[
    {
        "threshold": 0.3,
        "label": "train",
        "tp": 30,  # True Positives
        "fp": 20,  # False Positives
        "fn": 10,   # False Negatives
    },
    {
        "threshold": 0.4,
        "label": "train",
        "tp": 40,
        "fp": 20,
        "fn": 0,
    },
    ...
]
```

### Uploading Thresholded Results

After generating your thresholded object, you can upload it to Kolena using the `upload_results` method:

```py
from kolena.dataset import upload_results

def generate_thresholded_object():
    # Generate thresholded object based on your raw evaluation data and logic
    ...

df_results["thresholded"] = generate_thresholded_object()
upload_results("dataset_name", "model_name", df_results, thresholded_fields=["thresholded"])
```

You can upload multiple thresholded object fields if needed.

## Setting Up Thresholded Object-Powered Metrics

You can add thresholded metrics via the "Thresholded Object" button on the Configure Metric Group page.
This option is available only after uploading the thresholded object.

[screenshot of the button]

[screenshot of the detailed configuration page]

Supported metrics include:

- [Precision](../../metrics/precision.md)
- [Recall](../../metrics/recall.md)
- [F1 Score](../../metrics/f1-score.md)
- [Average Precision](../../metrics/average-precision.md)

[screenshot of qs page with configured metrics]

## Setting Up Thresholded Object-Powered Plots

### Task Plots

#### Precision vs. Recall Plot

[screenshot of plot with configuration]

#### F1 Score vs. Threshold

[screenshot of plot with configuration]

### Advanced Plots

#### Aggregation at Different Thresholds

[screenshot of plot with configuration]
