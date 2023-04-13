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
from typing import List
from typing import Optional
from typing import Type
from urllib.parse import quote

import dacite
from pydantic.dataclasses import dataclass

from kolena._api.v1.generic import Workflow as API
from kolena._utils import krequests
from kolena._utils import log
from kolena._utils.state import kolena_initialized
from kolena.workflow import GroundTruth
from kolena.workflow import Inference
from kolena.workflow import TestSample
from kolena.workflow.ground_truth import _validate_ground_truth_type
from kolena.workflow.inference import _validate_inference_type
from kolena.workflow.test_sample import _validate_test_sample_type

_RESERVED_WORKFLOW_NAMES = {"fr", "classification", "detection", "keypoints"}


@dataclass(frozen=True)
class RemoteEvaluator:
    """
    Remote evaluator for generic workflows.
    """

    #: The name of the workflow
    workflow: str

    #: The name of the evaluator
    name: str

    #: The full qualified docker image name
    image: Optional[str] = None

    #: Time the remote evaluator is registered
    created: Optional[str] = None

    #: Sensitive data registered with the evaluator, only included if requested explicitly
    secret: Optional[str] = None


@dataclass(frozen=True)
class Workflow:
    """
    The definition of a workflow and its associated types.
    """

    #: The name of the workflow. Should be unique, meaningful, and human-readable.
    name: str

    #: The :class:`kolena.workflow.TestSample` type for the workflow, using one of the builtin test sample types
    #: (e.g. :class:`kolena.workflow.Image`) and extending as necessary with additional fields.
    test_sample_type: Type[TestSample]

    #: The custom :class:`kolena.workflow.GroundTruth` type for the workflow.
    ground_truth_type: Type[GroundTruth]

    #: The custom :class:`kolena.workflow.Inference` type for the workflow.
    inference_type: Type[Inference]

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", self.name.strip())
        if self.name == "":
            raise ValueError("invalid zero-length name provided")
        if self.name.lower() in _RESERVED_WORKFLOW_NAMES:
            raise ValueError(f"invalid reserved name '{self.name}' provided")

        _validate_test_sample_type(self.test_sample_type)
        _validate_ground_truth_type(self.test_sample_type, self.ground_truth_type)
        _validate_inference_type(self.test_sample_type, self.inference_type)

    # convenience method to register evaluator without command-line
    @kolena_initialized
    def register_evaluator(self, evaluator_name: str, image: str, secret: Optional[str] = None) -> None:
        """
        Register a docker image for the evaluator of the workflow. This enables metrics evaluation to be run in
        kolena platform with the given image.

        If the evaluator needs to use sensitive data (secret) during its computation, pass it in :param:secret. The
        content would be store securely in kolena platform. At runtime, the evaluator can access the
        content in environment variable `KOLENA_EVALUATOR_SECRET` as is.

        :param evaluator_name: name of the evaluator
        :param image: fully qualified docker image name, i.e. <repository-url>/<image>:<tag>
        :param secret: sensitive data as string
        """
        register_evaluator(workflow=self.name, evaluator_name=evaluator_name, image=image, secret=secret)

    @kolena_initialized
    def get_evaluator(self, evaluator_name: str) -> API.EvaluatorResponse:
        """Get the docker image registered for the evaluator"""

        response = krequests.get(f"{API.Path.EVALUATOR}/{quote(self.name)}/{quote(evaluator_name)}")
        krequests.raise_for_status(response)

        return dacite.from_dict(API.EvaluatorResponse, response.json())


@kolena_initialized
def register_evaluator(workflow: str, evaluator_name: str, image: str, secret: Optional[str] = None) -> None:
    """
    Register a docker image for the evaluator of the workflow. This enables metrics evaluation to be run in
    kolena platform with the given image.

    If the evaluator needs to use sensitive data during its computation, pass it in ``secret``. The content would be
    store securely in kolena platform. At runtime, the evaluator can access the
    content in environment variable `KOLENA_EVALUATOR_SECRET`.

    :param evaluator_name: name of the evaluator; must match evaluator name generated by image
    :param image: fully qualified docker image name, i.e. <repository-url>/<image>:<tag>
    :param secret: sensitive data in string format
    """

    response = krequests.post(
        API.Path.REGISTER_EVALUATOR,
        json=dict(workflow=workflow, image=image, name=evaluator_name, secret=secret),
    )
    krequests.raise_for_status(response)

    log.info(f"Image {image} successfully registered for evaluator {evaluator_name}.")


@kolena_initialized
def list_evaluators(workflow: str) -> List[RemoteEvaluator]:
    """
    List all evaluators registered for a given workflow.

    :param workflow: workflow name
    :return: list of registered evaluators
    """

    response = krequests.get(f"{API.Path.EVALUATOR}/{quote(workflow)}")
    krequests.raise_for_status(response)

    return [
        RemoteEvaluator(workflow=workflow, name=evaluator.name, image=evaluator.image, created=evaluator.created)
        for evaluator in dacite.from_dict(API.ListEvaluatorsResponse, response.json()).evaluators
    ]


@kolena_initialized
def get_evaluator(workflow: str, evaluator_name: str, include_secret: bool = False) -> RemoteEvaluator:
    """
    Get the latest evaluator registered for the ``workflow`` and ``evaluator_name``.

    :param workflow: workflow name
    :param evaluator_name: evaluator name
    :param include_secret: retrieve secret registered with the evaluator or not
    :return: remote evaluator
    """

    response = krequests.get(
        f"{API.Path.EVALUATOR}/{quote(workflow)}/{quote(evaluator_name)}",
        params={"include_secret": include_secret},
    )
    krequests.raise_for_status(response)

    result = dacite.from_dict(API.EvaluatorResponse, response.json())
    return RemoteEvaluator(
        workflow=workflow,
        name=result.name,
        image=result.image,
        created=result.created,
        secret=result.secret,
    )
