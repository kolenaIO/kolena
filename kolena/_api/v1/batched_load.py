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
from enum import Enum

from kolena._utils.pydantic_v1.dataclasses import dataclass


class BatchedLoad:
    class Path(str, Enum):
        INIT_UPLOAD = "/batched-load/upload/init"
        COMPLETE_DOWNLOAD = "/batched-load/download/complete"
        UPLOAD_SIGNED_URL_STUB = "/batched-load/upload/signed-url"
        DOWNLOAD_BY_PATH_STUB = "/batched-load/download/by-path"

        @classmethod
        def upload_signed_url(cls, load_uuid: str) -> str:
            return f"{cls.UPLOAD_SIGNED_URL_STUB.value}/{load_uuid}"

        @classmethod
        def download_by_path(cls, path: str) -> str:
            return f"{cls.DOWNLOAD_BY_PATH_STUB.value}/{path}"

    @dataclass(frozen=True)
    class WithLoadUUID:
        uuid: str

    @dataclass(frozen=True)
    class SignedURL:
        signed_url: str

    @dataclass(frozen=True)
    class BaseInitDownloadRequest:
        batch_size: int

    @dataclass(frozen=True)
    class InitDownloadPartialResponse(WithLoadUUID):
        path: str

    @dataclass(frozen=True)
    class CompleteDownloadRequest(WithLoadUUID):
        ...

    @dataclass(frozen=True)
    class InitiateUploadResponse(WithLoadUUID):
        ...
