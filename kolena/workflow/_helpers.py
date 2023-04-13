from typing import cast
from typing import Tuple
from typing import Type

from pydantic import validate_arguments

from kolena._utils.validators import ValidatorConfig
from kolena.workflow import GroundTruth
from kolena.workflow import Inference
from kolena.workflow import Model
from kolena.workflow import TestCase
from kolena.workflow import TestSample
from kolena.workflow import TestSuite
from kolena.workflow import Workflow


@validate_arguments(config=ValidatorConfig)
def define_workflow(
    name: str,
    test_sample_type: Type[TestSample],
    ground_truth_type: Type[GroundTruth],
    inference_type: Type[Inference],
) -> Tuple[Workflow, Type[TestCase], Type[TestSuite], Type[Model]]:
    """
    Define a new workflow, specifying its test sample, ground truth, and inference types.

    Provided as a convenience method to create the :class:`TestCase`, :class:`TestSuite`, and :class:`Model` objects
    for a new workflow. These objects can also be defined manually by subclassing them and binding the ``workflow``
    class variable:

    .. code-block:: python

        from kolena.workflow import TestCase
        from my_code import my_workflow

        class MyTestCase(TestCase):
            workflow = my_workflow

    :return: the :class:`Workflow` object for this workflow along with the :class:`TestCase`, :class:`TestSuite`,
        and :class:`Model` objects to use when creating and running tests for this workflow.
    """
    workflow = Workflow(
        name=name,
        test_sample_type=test_sample_type,
        ground_truth_type=ground_truth_type,
        inference_type=inference_type,
    )

    test_case = type("TestCase", (TestCase,), {"workflow": workflow})
    test_suite = type("TestSuite", (TestSuite,), {"workflow": workflow, "_test_case_type": test_case})
    model = type("Model", (Model,), {"workflow": workflow})

    return workflow, cast(Type[TestCase], test_case), cast(Type[TestSuite], test_suite), cast(Type[Model], model)
