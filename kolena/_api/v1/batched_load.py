from enum import Enum

from pydantic.dataclasses import dataclass


class BatchedLoad:
    class Path(str, Enum):
        INIT_UPLOAD = "/batched-load/upload/init"
        COMPLETE_DOWNLOAD = "/batched-load/download/complete"
        UPLOAD_SIGNED_URL_STUB = "/batched-load/upload/signed-url"
        DOWNLOAD_BY_PATH_STUB = "/batched-load/download/by-path"

        @classmethod
        def upload_signed_url(cls, load_uuid: str) -> str:
            return f"{cls.UPLOAD_SIGNED_URL_STUB}/{load_uuid}"

        @classmethod
        def download_by_path(cls, path: str) -> str:
            return f"{cls.DOWNLOAD_BY_PATH_STUB}/{path}"

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
