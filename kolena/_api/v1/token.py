from dataclasses import dataclass


@dataclass(frozen=True)
class ValidateRequest:
    api_token: str
    version: str


@dataclass(frozen=True)
class ValidateResponse:
    tenant: str
    access_token: str
    token_type: str
    tenant_telemetry: bool
