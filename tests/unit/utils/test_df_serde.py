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
from typing import Any

import numpy as np
import pytest

from kolena._utils.serde import deserialize_embedding_vector
from kolena._utils.serde import serialize_embedding_vector


@pytest.mark.parametrize(
    "dtype",
    [str, np.int32, np.int64, np.complex128, "b", "B", ">H", "<f", "d", "i4", "u4", "f8", "c16", "a25", "U25"],
)
def test__embedding_vector__serde(dtype: Any) -> None:
    want = np.array([[[1.0, 2.2], [3 + 1 / 3, 4]], [[5, 6], [7, 8]]]).astype(dtype)
    serialized = serialize_embedding_vector(want)
    got = deserialize_embedding_vector(serialized)
    assert np.array_equal(got, want)
    assert got.dtype == want.dtype
