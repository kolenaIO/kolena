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
from typing import Set

from kolena.workflow.annotation import ScoredLabeledPolygon


def filter_inferences(
    inferences: List[ScoredLabeledPolygon],
    confidence_score: Optional[float] = None,
    labels: Optional[Set[str]] = None,
) -> List[ScoredLabeledPolygon]:
    filtered_by_confidence = (
        [inf for inf in inferences if inf.score >= confidence_score] if confidence_score else inferences
    )
    if labels is None:
        return filtered_by_confidence
    return [inf for inf in filtered_by_confidence if inf.label in labels]
