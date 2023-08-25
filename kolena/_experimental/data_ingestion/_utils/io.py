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
import json
from typing import Any

import requests


def load_json(path: str) -> Any:
    if path.startswith("http://") or path.startswith("https://"):
        response = requests.get(path)
        response.raise_for_status()
        data = response.json()
    elif path.startswith("s3://"):
        import boto3

        s3_bucket, s3_key = path[5:].split("/", 1)
        s3 = boto3.client("s3")
        response = s3.get_object(Bucket=s3_bucket, Key=s3_key)
        data = json.load(response["Body"])  # type: ignore
    else:
        with open(path) as file:
            data = json.load(file)

    return data


def load_csv() -> None:
    ...
