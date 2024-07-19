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
import uuid
from unittest.mock import Mock
from unittest.mock import patch

from kolena._experimental.trace.trace import _Trace
from kolena._experimental.trace.trace import kolena_trace


@patch.object(_Trace, "_push_data")
def test__kolena_trace(mock_push_data: Mock) -> None:
    mock_push_data.return_value = None
    dataset_name = "test_kolena_trace_provided_id" + uuid.uuid4().hex
    model_name = "test_kolena_trace_provided_id_model" + uuid.uuid4().hex

    @kolena_trace(dataset_name=dataset_name, id_fields=["request_id"], record_timestamp=False)
    def predict(data, request_id):
        return data + request_id

    result = predict(1, 2)
    assert result == 3
    assert predict.__name__ == "predict"
    assert predict.dataset_name == dataset_name
    assert predict.model_name == f"{dataset_name}_model"
    assert predict.model_name_field is None
    assert predict.id_fields == ["request_id"]

    @kolena_trace
    def predict(data, request_id):
        return data + request_id

    result = predict(1, 2)
    assert result == 3
    assert predict.__name__ == "predict"
    assert predict.dataset_name == "predict"
    assert predict.model_name == "predict_model"
    assert predict.model_name_field is None
    assert predict.id_fields == ["_kolena_id"]

    def predict(data, request_id):
        return data + request_id

    predict = kolena_trace(predict)

    result = predict(1, 2)
    assert result == 3
    assert predict.__name__ == "predict"
    assert predict.dataset_name == "predict"
    assert predict.model_name == "predict_model"
    assert predict.model_name_field is None
    assert predict.id_fields == ["_kolena_id"]

    @kolena_trace(model_name=model_name)
    def predict(data, request_id):
        return data + request_id

    result = predict(1, 2)
    assert result == 3
    assert predict.__name__ == "predict"
    assert predict.dataset_name == "predict"
    assert predict.model_name == model_name
    assert predict.model_name_field is None
    assert predict.id_fields == ["_kolena_id"]

    @kolena_trace(model_name=model_name, model_name_field="model_name")
    def predict(data, request_id, model_name):
        return data + request_id

    result = predict(1, 2, model_name=model_name)
    assert result == 3
    assert predict.__name__ == "predict"
    assert predict.dataset_name == "predict"
    assert predict.model_name == model_name
    assert predict.model_name_field == "model_name"
    assert predict.id_fields == ["_kolena_id"]
