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
from abc import ABCMeta
from abc import abstractmethod
from typing import Any
from typing import Dict

from kolena._api.v1.detection import Metrics
from kolena._utils.frozen import Frozen
from kolena._utils.serializable import Serializable


class TestConfig(Frozen, Serializable, metaclass=ABCMeta):
    """
    Base class for a testing configuration.

    See concrete implementations :class:`kolena.detection.config.FixedThreshold`,
    :class:`kolena.detection.config.F1Optimal` for details.
    """

    @abstractmethod
    def __init__(self) -> None:
        ...

    @abstractmethod
    def _to_run_config(self) -> Metrics.RunConfig:
        ...

    def _to_dict(self) -> Dict[str, Any]:
        run_config = self._to_run_config()
        return {k: v for k, v in run_config.__dict__.items() if not k.startswith("_")}
