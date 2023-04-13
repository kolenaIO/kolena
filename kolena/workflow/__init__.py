# noreorder
from ._datatypes import DataObject
from .test_sample import Metadata
from .test_sample import Image
from .test_sample import ImagePair
from .test_sample import ImageText
from .test_sample import TestSample
from .test_sample import Composite
from .test_sample import Text
from .test_sample import BaseVideo
from .test_sample import Video
from .ground_truth import GroundTruth
from .inference import Inference
from .workflow import Workflow
from .test_case import TestCase
from .test_suite import TestSuite
from .model import Model
from .evaluator import AxisConfig
from .evaluator import Plot
from .evaluator import Curve
from .evaluator import CurvePlot
from .evaluator import ConfusionMatrix
from .evaluator import Histogram
from .evaluator import MetricsTestCase
from .evaluator import MetricsTestSample
from .evaluator import MetricsTestSuite
from .evaluator import Evaluator
from .evaluator import EvaluatorConfiguration
from .evaluator_function import BasicEvaluatorFunction as _BasicEvaluatorFunction
from .evaluator_function import TestCases
from .evaluator_function import EvaluationResults
from .test_run import TestRun
from .test_run import test
from ._helpers import define_workflow

__all__ = [
    "DataObject",
    "Metadata",
    "Image",
    "ImagePair",
    "ImageText",
    "TestSample",
    "Composite",
    "Text",
    "BaseVideo",
    "Video",
    "GroundTruth",
    "Inference",
    "Workflow",
    "TestCase",
    "TestSuite",
    "Model",
    "AxisConfig",
    "Plot",
    "Curve",
    "CurvePlot",
    "ConfusionMatrix",
    "Histogram",
    "MetricsTestCase",
    "MetricsTestSample",
    "MetricsTestSuite",
    "Evaluator",
    "EvaluatorConfiguration",
    "BasicEvaluatorFunction",
    "TestCases",
    "EvaluationResults",
    "TestRun",
    "test",
    "define_workflow",
]

# """Copied directly from evaluator_function.py for documentation purposes."""
#: ``kolena.workflow.BasicEvaluatorFunction`` introduces a function based evaluator implementation that takes
#: the inferences for all test samples in a test suite and a :class:`kolena.workflow.TestCases` as input, and computes
#: the corresponding test-sample-level, test-case-level, and test-suite-level metrics (and optionally plots) as output.
#:
#: The control flow is in general more streamlined than with :class:`kolena.workflow.Evaluator`, but requires a couple
#: of assumptions to hold:
#:
#: - Test-sample-level metrics do not vary by test case
#: - Ground truths corresponding to a given test sample do not vary by test case
#:
#: This ``BasicEvaluatorFunction`` is provided to the test run at runtime, and is expected to have the
#: following signature:
#:
#: :param List[kolena.workflow.TestSample] test_samples: A list of distinct :class:`kolena.workflow.TestSample` values
#:     that correspond to all test samples in the test run.
#: :param List[kolena.workflow.GroundTruth] ground_truths: A list of :class:`kolena.workflow.GroundTruth` values
#:     corresponding to and sequenced in the same order as ``test_samples``.
#: :param List[kolena.workflow.Inference] inferences: A list of :class:`kolena.workflow.Inference` values corresponding
#:     to and sequenced in the same order as ``test_samples``.
#: :param TestCases test_cases: An instance of :class:`kolena.workflow.TestCases`, generally used to provide iteration
#:        groupings for evaluating test-case-level metrics.
#: :param EvaluatorConfiguration evaluator_configuration: The configuration to use when performing the evaluation.
#:     This parameter may be omitted in the function definition if running with no configuration.
#: :rtype: :class:`kolena.workflow.EvaluationResults`
#: :return: An object tracking the test-sample-level, test-case-level and test-suite-level metrics and plots for the
#:     input collection of test samples.
BasicEvaluatorFunction = _BasicEvaluatorFunction
