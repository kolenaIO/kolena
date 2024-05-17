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
from kolena._api.v2.quality_standard import Path as QualityStandardPath
from kolena._utils import krequests_v2 as krequests
from kolena.dataset.dataset import _load_dataset_metadata


def create_quality_standard(dataset_name: str, quality_standard: dict) -> None:
    dataset = _load_dataset_metadata(dataset_name)

    response = krequests.put(
        QualityStandardPath.QUALITY_STANDARD,
        params=dict(dataset_id=dataset.id),
        json=quality_standard,
        api_version="v2",
    )
    krequests.raise_for_status(response)
