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
from pydantic.dataclasses import dataclass

from kolena.workflow import define_workflow
from kolena.workflow import EvaluatorConfiguration
from kolena.workflow import GroundTruth
from kolena.workflow import Inference
from kolena.workflow import Text

workflow, TestCase, TestSuite, Model = define_workflow("Retrieval Augmented Generation", Text, GroundTruth, Inference)


@dataclass(frozen=True)
class Configuration(EvaluatorConfiguration):
    # leverage DataObject __str__
    def display_name(self) -> str:
        return self.__str__()
