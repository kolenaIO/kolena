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
"""
The Multiclass Classification pre-built workflow provides out-of-the-box definitions for testing image-based multiclass
classification models on Kolena.

Example open-source datasets that can be tested with this workflow:

<div class="grid cards" markdown>
- [CIFAR-10 and CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html)

    ---

    <figure markdown>
        ![Tiny 32x32 pixel color images with 10 and 100 classes](../../assets/images/cifar.png)
        <figcaption>Tiny 32x32 pixel color images with 10 and 100 classes</figcaption>
    </figure>

- [MNIST database](https://paperswithcode.com/dataset/mnist)

    ---

    <figure markdown>
        ![Single-channel 28x28 pixel handwritten digits (0-9)](../../assets/images/mnist.png)
        <figcaption>
            Single-channel 28x28 pixel handwritten digits (0-9)
            ([attribution](https://commons.wikimedia.org/wiki/File:MnistExamplesModified.png))
        </figcaption>
    </figure>
</div>
"""

# noreorder
from .workflow import GroundTruth
from .workflow import Inference
from .workflow import TestCase
from .workflow import TestSuite
from .workflow import Model
from .workflow import PerImageMetrics
from .workflow import PerClassMetrics
from .workflow import AggregateMetrics
from .workflow import TestSuiteMetrics
from .workflow import ThresholdConfiguration
from .evaluator import evaluate_multiclass_classification
from .test_run import test

__all__ = [
    "GroundTruth",
    "Inference",
    "TestCase",
    "TestSuite",
    "Model",
    "PerImageMetrics",
    "PerClassMetrics",
    "AggregateMetrics",
    "TestSuiteMetrics",
    "ThresholdConfiguration",
    "evaluate_multiclass_classification",
    "test",
]
