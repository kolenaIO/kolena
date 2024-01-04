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
import json
from base64 import b64decode
from base64 import b64encode
from enum import Enum
from io import BytesIO
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Type
from typing import TypeVar
from typing import Union

import dacite
import numpy as np
import pandas as pd


def serialize_embedding_vector(embedding_vector: np.ndarray) -> str:
    memfile = BytesIO()
    np.save(memfile, embedding_vector, allow_pickle=False)
    memfile.seek(0)
    embedding_b64_bytes = b64encode(memfile.read())
    return embedding_b64_bytes.decode("utf-8")


def deserialize_embedding_vector(b64blob: str) -> np.ndarray:
    embedding_bytes = b64decode(b64blob)
    memfile = BytesIO()
    memfile.write(embedding_bytes)
    memfile.seek(0)
    return np.load(memfile)


def as_float64_array(maybe_arr: Optional[List[Union[float, int]]]) -> Optional[np.ndarray]:
    return np.array(maybe_arr).astype(np.float64) if maybe_arr is not None else None


def as_serialized_json(maybe_json: Optional[Union[np.ndarray, Any]]) -> Optional[str]:
    if isinstance(maybe_json, np.ndarray):
        maybe_json = maybe_json.tolist()
    return json.dumps(maybe_json, sort_keys=True) if maybe_json is not None else None


def as_deserialized_json(maybe_json_string: Optional[str]) -> Optional[Any]:
    return json.loads(maybe_json_string) if maybe_json_string is not None else None


def with_serialized_columns(df: pd.DataFrame, object_columns: List[str]) -> pd.DataFrame:
    df_serializable = df.copy()
    for col in object_columns:
        df_serializable[col] = df_serializable[col].apply(as_serialized_json)
    return df_serializable


T = TypeVar("T")
_CONFIG = dacite.Config(cast=[Enum, tuple], check_types=False)


def from_dict(data_class: Type[T], data: Dict[str, Any]) -> T:
    return dacite.from_dict(data_class=data_class, data=data, config=_CONFIG)
