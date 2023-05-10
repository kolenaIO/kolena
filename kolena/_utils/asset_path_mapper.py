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
from kolena._api.v1.fr import Asset


class AssetPathMapper:
    """
    Maps images associated with test run as PNG assets to the configured S3 bucket as:
        s3://<bucket>/<database>/<schema>/<test-run-id>/<image-id>/<key>.png
    """

    def __init__(self, config: Asset.Config):
        self.bucket = config.bucket
        self.prefix = config.prefix

    def absolute_locator(self, test_run_id: int, load_uuid: str, image_id: int, key: str) -> str:
        return self._absolute_locator(self.relative_locator(self.path_stub(test_run_id, load_uuid, image_id, key)))

    def relative_locator(self, path_stub: str) -> str:
        return f"{self.prefix}/{path_stub}"

    @staticmethod
    def path_stub(test_run_id: int, load_uuid: str, image_id: int, key: str) -> str:
        return f"{test_run_id}/{image_id}/{key}-{load_uuid}.png"

    def _absolute_locator(self, relative_locator: str) -> str:
        relative_locator_stripped = relative_locator.strip("/")
        return f"s3://{self.bucket}/{relative_locator_stripped}"
