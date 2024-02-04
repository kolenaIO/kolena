# Copyright 2021-2024 Kolena Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


def accuracy(true_positives: int, false_positives: int, false_negatives: int, true_negatives: int) -> float:
    """
    Accuracy represents the proportion of inferences that are correct (including both positives and negatives).

    $$
    \\text{Accuracy} = \\frac{\\text{# TP} + \\text{# TN}}
    {\\text{# TP} + \\text{# FP} + \\text{# FN} + \\text{# TN}}
    $$

    <div class="grid cards" markdown>
    - :kolena-metrics-glossary-16: Metrics Glossary: [Accuracy ↗](../metrics/accuracy.md)
    </div>

    :param true_positives: Number of true positive inferences.
    :param false_positives: Number of false positive inferences.
    :param false_negatives: Number of false negatives.
    :param true_negatives: Number of true negatives.
    """
    numerator = true_positives + true_negatives
    denominator = true_positives + false_positives + false_negatives + true_negatives
    return numerator / denominator if denominator > 0 else 0


def precision(true_positives: int, false_positives: int) -> float:
    """
    Precision represents the proportion of inferences that are correct.

    $$
    \\text{Precision} = \\frac{\\text{# True Positives}}{\\text{# True Positives} + \\text{# False Positives}}
    $$

    <div class="grid cards" markdown>
    - :kolena-metrics-glossary-16: Metrics Glossary: [Precision ↗](../metrics/precision.md)
    </div>

    :param true_positives: Number of true positive inferences.
    :param false_positives: Number of false positive inferences.
    """
    denominator = true_positives + false_positives
    return true_positives / denominator if denominator > 0 else 0


def recall(true_positives: int, false_negatives: int) -> float:
    """
    Recall (TPR or sensitivity) represents the proportion of ground truths that were successfully predicted.

    $$
    \\text{Recall} = \\frac{\\text{# True Positives}}{\\text{# True Positives} + \\text{# False Negatives}}
    $$

    <div class="grid cards" markdown>
    - :kolena-metrics-glossary-16: Metrics Glossary: [Recall ↗](../metrics/recall.md)
    </div>

    :param true_positives: Number of true positive inferences.
    :param false_negatives: Number of false negatives.
    """
    denominator = true_positives + false_negatives
    return true_positives / denominator if denominator > 0 else 0


def f1_score(true_positives: int, false_positives: int, false_negatives: int) -> float:
    """
    F<sub>1</sub>-score is the harmonic mean between [`precision`][kolena.metrics.precision] and
    [`recall`][kolena.metrics.recall].

    $$
    \\begin{align}
    \\text{F1}
    &= \\frac{2}{\\frac{1}{\\text{Precision}} + \\frac{1}{\\text{Recall}}} \\\\[1em]
    &= 2 \\times \\frac{\\text{Precision} \\times \\text{Recall}}{\\text{Precision} + \\text{Recall}}
    \\end{align}
    $$

    <div class="grid cards" markdown>
    - :kolena-metrics-glossary-16: Metrics Glossary: [F<sub>1</sub>-score ↗](../metrics/f1-score.md)
    </div>

    :param true_positives: Number of true positive inferences.
    :param false_positives: Number of false positive inferences.
    :param false_negatives: Number of false negatives.
    """
    prec = precision(true_positives, false_positives)
    rec = recall(true_positives, false_negatives)
    denominator = prec + rec
    return 2 * prec * rec / denominator if denominator > 0 else 0


def fpr(true_negatives: int, false_positives: int) -> float:
    """
    False positive rate represents the proportion of negative ground truths that were incorrectly predicted as positive
    by the model.

    $$
    \\text{FPR} = \\frac{\\text{# False Positives}}{\\text{# False Positives} + \\text{# True Negatives}}
    $$

    <div class="grid cards" markdown>
    - :kolena-metrics-glossary-16: Metrics Glossary: [False Positive Rate ↗](../metrics/fpr.md)
    </div>

    :param true_negatives: Number of true negatives.
    :param false_positives: Number of false positives.
    """
    denominator = true_negatives + false_positives
    return false_positives / denominator if denominator > 0 else 0


def specificity(true_negatives: int, false_positives: int) -> float:
    """
    Specificity (TNR) represents the proportion of negative ground truths that were correctly predicted.

    $$
    \\text{Specificity} = \\frac{\\text{# True Negatives}}{\\text{# True Negatives} + \\text{# False Positives}}
    $$

    <div class="grid cards" markdown>
    - :kolena-metrics-glossary-16: Metrics Glossary: [Specificity ↗](../metrics/specificity.md)
    </div>

    :param true_negatives: Number of true negatives.
    :param false_positives: Number of false positives.
    """
    denominator = true_negatives + false_positives
    return true_negatives / denominator if denominator > 0 else 0
