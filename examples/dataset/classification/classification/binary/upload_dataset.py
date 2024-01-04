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
from typing import Dict

import pandas as pd
from classification.binary.constants import BUCKET
from classification.binary.constants import DATASET

import kolena
from kolena.dataset import register_dataset
from kolena.workflow._datatypes import _get_full_type
from kolena.workflow._datatypes import _serialize_dataobject
from kolena.workflow.annotation import ClassificationLabel
from kolena.workflow.io import _dataframe_object_serde


def to_serialized_dataframe(df: pd.DataFrame, column: str) -> pd.DataFrame:
    result = _dataframe_object_serde(df, _serialize_dataobject)
    result[column] = result.to_dict("records")
    return result[[column]]


def to_label_object(x: str) -> Dict[str, str]:
    label = ClassificationLabel(x)
    return {
        "label": label.label,
        "data_type": _get_full_type(label),
    }


def main() -> None:
    kolena.initialize(verbose=True)
    df_metadata = pd.read_csv(f"s3://{BUCKET}/{DATASET}/raw/{DATASET}.csv", storage_options={"anon": True})
    id_fields = ["locator"]
    ground_truth_field = "label"

    df_serialized_metadata = to_serialized_dataframe(df_metadata[["width", "height"]], column="metadata")
    df_serialized_ground_truth = df_metadata[ground_truth_field].apply(lambda x: to_label_object(x))
    df_datapoints = pd.concat([df_metadata[id_fields], df_serialized_metadata, df_serialized_ground_truth], axis=1)
    register_dataset(DATASET, df_datapoints, id_fields)


if __name__ == "__main__":
    main()
