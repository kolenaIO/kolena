from typing import Type

import pytest

from kolena.workflow import GroundTruth
from kolena.workflow import GroundTruth as BaseGroundTruth
from kolena.workflow import Inference
from kolena.workflow import Inference as BaseInference
from kolena.workflow import TestSample
from kolena.workflow import TestSample as BaseTestSample


@pytest.fixture
def test_sample_type() -> Type[TestSample]:
    class ExampleTestSample(BaseTestSample):
        ...

    return ExampleTestSample


@pytest.fixture
def ground_truth_type() -> Type[GroundTruth]:
    class ExampleGroundTruth(BaseGroundTruth):
        ...

    return ExampleGroundTruth


@pytest.fixture
def inference_type() -> Type[Inference]:
    class ExampleInference(BaseInference):
        ...

    return ExampleInference
