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
from typing import List
from typing import Optional
from typing import Type
from urllib.parse import quote

import dacite

from kolena._api.v1.generic import Workflow as API
from kolena._utils import krequests
from kolena._utils import log
from kolena._utils.pydantic_v1.dataclasses import dataclass
from kolena._utils.state import kolena_initialized
from kolena.workflow import GroundTruth
from kolena.workflow import Inference
from kolena.workflow import TestSample
from kolena.workflow.ground_truth import _validate_ground_truth_type
from kolena.workflow.inference import _validate_inference_type
from kolena.workflow.test_sample import _validate_test_sample_type

_RESERVED_WORKFLOW_NAMES = {"fr", "classification", "detection", "keypoints"}


@dataclass(frozen=True)
class EvaluatorRoleConfig:
    #: evaluator role arn
    job_role_arn: str

    #: external_id to be used when assume_role
    external_id: str

    #: role_arn to be assume-into
    assume_role_arn: str


@dataclass(frozen=True)
class RemoteEvaluator:
    """
    Remote evaluator for workflows built with ``kolena.workflow``.
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

    #: AWS assume role configurations for the evaluator job if configured
    aws_role_config: Optional[EvaluatorRoleConfig] = None

    @classmethod
    def _from_api_response(cls, workflow: str, response: API.EvaluatorResponse) -> "RemoteEvaluator":
        aws_role_config = None
        if response.aws_role_config:
            aws_role_config = EvaluatorRoleConfig(
                job_role_arn=response.aws_role_config.job_role_arn,
                external_id=response.aws_role_config.external_id,
                assume_role_arn=response.aws_role_config.assume_role_arn,
            )
        return cls(
            workflow=workflow,
            name=response.name,
            image=response.image,
            created=response.created,
            secret=response.secret,
            aws_role_config=aws_role_config,
        )


@dataclass(frozen=True)
class Workflow:
    """The definition of a workflow and its associated types."""

    name: str
    """The name of the workflow. Should be unique, meaningful, and human-readable."""

    test_sample_type: Type[TestSample]
    """The [`TestSample`][kolena.workflow.TestSample] type for the workflow, using one of the builtin test sample types
    (e.g. [`Image`][kolena.workflow.Image]) and extending as necessary with additional fields."""

    ground_truth_type: Type[GroundTruth]
    """The custom [`GroundTruth`][kolena.workflow.GroundTruth] type for the workflow."""

    inference_type: Type[Inference]
    """The custom [`Inference`][kolena.workflow.Inference] type for the workflow."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", self.name.strip())
        if self.name == "":
            raise ValueError("invalid zero-length name provided")
        if self.name.lower() in _RESERVED_WORKFLOW_NAMES:
            raise ValueError(f"invalid reserved name '{self.name}' provided")

        _validate_test_sample_type(self.test_sample_type)
        _validate_ground_truth_type(self.test_sample_type, self.ground_truth_type)
        _validate_inference_type(self.test_sample_type, self.inference_type)

    @kolena_initialized
    def register_evaluator(
        self,
        evaluator_name: str,
        image: str,
        secret: Optional[str] = None,
        aws_assume_role: Optional[str] = None,
    ) -> RemoteEvaluator:
        """
        Convenience method to register evaluator for the workflow.

        See [`register_evaluator`][kolena.workflow.workflow.register_evaluator] for details.
        """
        return register_evaluator(
            workflow=self.name,
            evaluator_name=evaluator_name,
            image=image,
            secret=secret,
            aws_assume_role=aws_assume_role,
        )

    @kolena_initialized
    def get_evaluator(self, evaluator_name: str, include_secret: bool = False) -> RemoteEvaluator:
        """
        Get the docker image registered for the evaluator.

        See [`get_evaluator`][kolena.workflow.workflow.get_evaluator] for details.
        """

        return get_evaluator(self.name, evaluator_name, include_secret)


@kolena_initialized
def register_evaluator(
    workflow: str,
    evaluator_name: str,
    image: str,
    secret: Optional[str] = None,
    aws_assume_role: Optional[str] = None,
) -> RemoteEvaluator:
    """
    Register a docker image for the evaluator of the workflow. This enables metrics evaluation to be run in
    kolena platform with the given image.

    If the evaluator needs to use sensitive data during its computation, pass it in ``secret``. The content would be
    store securely in kolena platform. At runtime, the evaluator can access the content in environment variable
    `KOLENA_EVALUATOR_SECRET`.

    If the evaluator needs to interact with AWS APIs, specify the role it would use in ``aws_assume_role``. The
    runtime would be executed by a Kolena-side AWS role that is configured to assume the ``aws_assume_role`` using a
    randomly generated `external_id`. The AWS role arn from Kolena and the `external_id` would be returned in the
    response `aws_assume_role` field. The only configuration on user side is to add Kolena AWS role to
    ``aws_assume_role``'s trust policy. At runtime, the evaluator code can get the external_id in environment
    variable `KOLENA_EVALUATOR_EXTERNAL_ID` and the ``aws_assume_role`` would be passed in
    `KOLENA_EVALUATOR_ASSUME_ROLE_ARN`.

    An example to assume role at evaluator runtime is shown below::

        external_id = os.environ["KOLENA_EVALUATOR_EXTERNAL_ID"]
        assume_role_arn = os.environ["KOLENA_EVALUATOR_ASSUME_ROLE_ARN"]  # this is the role to assume

        sts = boto3.client("sts")
        response = sts.assume_role(
            RoleArn=assume_role_arn,
            RoleSessionName="metrics-evaluator",
            ExternalId=external_id,
        )
        credentials = response["Credentials"]
        client = boto3.client(
            "s3",
            aws_access_key_id=credentials["AccessKeyId"],
            aws_secret_access_key=credentials["SecretAccessKey"],
            aws_session_token=credentials["SessionToken"],
        )
        buckets = client.list_buckets()

    :param evaluator_name: name of the evaluator; must match evaluator name generated by image
    :param image: fully qualified docker image name, i.e. <repository-url>/<image>:<tag>
    :param secret: sensitive data in string format
    :param aws_assume_role: AWS role that the evaluator would use to access AWS APIs
    :return: registered evaluator data
    """

    response = krequests.post(
        API.Path.EVALUATOR.value,
        json=dict(workflow=workflow, image=image, name=evaluator_name, secret=secret, aws_assume_role=aws_assume_role),
    )
    krequests.raise_for_status(response)

    log.info(f"Image {image} successfully registered for evaluator {evaluator_name}.")

    result = dacite.from_dict(API.EvaluatorResponse, response.json())
    return RemoteEvaluator._from_api_response(workflow, result)


@kolena_initialized
def list_evaluators(workflow: str) -> List[RemoteEvaluator]:
    """
    List all evaluators registered for a given workflow.

    :param workflow: workflow name
    :return: list of registered evaluators
    """

    response = krequests.get(f"{API.Path.EVALUATOR.value}/{quote(workflow)}")
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
        endpoint_path=f"{API.Path.EVALUATOR.value}/{quote(workflow)}/{quote(evaluator_name)}",
        params=dict(include_secret=include_secret),
    )
    krequests.raise_for_status(response)

    result = dacite.from_dict(API.EvaluatorResponse, response.json())
    return RemoteEvaluator._from_api_response(workflow, result)
