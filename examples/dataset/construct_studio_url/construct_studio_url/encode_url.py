# Copyright 2021-2025 Kolena Inc.
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
from urllib.parse import quote

import lzstring


def construct_studio_url(tenant: str, dataset_id: int, datapoint_field: str, value: str) -> str:
    """
    Generate a Kolena dataset studio URL with a filter applied.

    Args:
        tenant: The tenant name (e.g., "try")
        dataset_id: The ID of the dataset (e.g., 59)
        datapoint_field: The field name to filter on (e.g., "uuid")
        value: The value to filter for (e.g., "abc")
    Returns:
        A fully formatted URL string with the filter encoded
    """
    field_name = f"datapoint.{datapoint_field}"
    filter_data = {
        "contains": value,
        "options": {
            "regex_match": False,
            "regex_flags": "",
            "inverse": False,
        },
    }
    json_string = json.dumps(filter_data, separators=(",", ":"))

    compressor = lzstring.LZString()
    compressed = compressor.compressToEncodedURIComponent(json_string)
    filter_string = f"{field_name}:{compressed}:c"
    encoded_filter = quote(filter_string, safe="")

    # Step 7: Build the complete URL
    base_url = f"https://app.kolena.com/{tenant}/dataset/studio"
    full_url = f"{base_url}?datasetId={dataset_id}&filters={encoded_filter}"

    return full_url
