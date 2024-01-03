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
from functools import partial

from kolena._utils import krequests
from kolena._utils.state import API_V2

get = partial(krequests.get, api_version=API_V2)
post = partial(krequests.post, api_version=API_V2)
put = partial(krequests.put, api_version=API_V2)
delete = partial(krequests.delete, api_version=API_V2)
raise_for_status = krequests.raise_for_status
