# Copyright 2021-2023 Kolena Inc.
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


def precision(true_positives: int, false_positives: int) -> float:
    """
    Precision score following the formula:

    $$
    \\text{Precision} = \\frac{\\text{# True Positives}}{\\text{# True Positives} + \\text{# False Positives}}
    $$

    :param true_positives: Number of true positive predictions.
    :param false_positives: Number of false positive predictions.
    """
    return true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0


def recall(true_positives: int, false_negatives: int) -> float:
    """
    Recall score following the formula:

    $$
    \\text{Recall} = \\frac{\\text{# True Positives}}{\\text{# True Positives} + \\text{# False Negatives}}
    $$

    :param true_positives: Number of true positive predictions.
    :param false_negatives: Number of false negatives.
    """
    return true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0


def f1(true_positives: int, false_positives: int, false_negatives: int) -> float:
    """
    F1 score following the formula:

    $$
    \\text{F1} = 2 \\times \\frac{\\text{Precision} \\times \\text{Recall}}{\\text{Precision} + \\text{Recall}}
    $$

    :param true_positives: Number of true positive predictions.
    :param false_positives: Number of false positive predictions.
    :param false_negatives: Number of false negatives.
    """
    prec = precision(true_positives, false_positives)
    rec = recall(true_positives, false_negatives)
    return (2 * prec * rec) / (prec + rec)
