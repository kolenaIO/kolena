from random import random

from keypoint_detection.workflow import TestCaseMetrics
from keypoint_detection.workflow import TestSampleMetrics

from kolena.workflow import Evaluator


class KeypointsEvaluator(Evaluator):
    """See API documentation for details."""

    def compute_test_case_metrics(self, test_case, inferences, metrics, configuration=None):
        # Generate dummy metrics for demo purposes.
        return TestCaseMetrics(random(), random())

    def compute_test_sample_metrics(self, test_case, inferences, configuration=None):
        # Generate dummy metrics for demo purposes.
        return [(i[0], TestSampleMetrics(random(), bool(random() > 0.5))) for i in inferences]
