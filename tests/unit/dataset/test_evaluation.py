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
import pytest

from kolena.dataset.evaluation import _validate_tags
from kolena.errors import IncorrectUsageError


@pytest.mark.parametrize(
    "tags, has_error",
    [([], False), (["tag-a", "tag-b"], False), (["tag-a", " "], True), ([""], True)],
)
def test__validate_tags(tags: list[str], has_error: bool) -> None:
    if has_error:
        with pytest.raises(IncorrectUsageError):
            _validate_tags(tags)
    else:
        _validate_tags(tags)
