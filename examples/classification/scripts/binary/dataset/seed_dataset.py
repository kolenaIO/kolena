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
import sys
from argparse import ArgumentParser
from argparse import Namespace
from typing import Dict

import pandas as pd

import kolena
from kolena._experimental.dataset._dataset import register_dataset
from kolena.workflow._datatypes import _get_full_type
from kolena.workflow._datatypes import _serialize_dataobject
from kolena.workflow.annotation import ClassificationLabel
from kolena.workflow.io import _dataframe_object_serde

BUCKET = "kolena-public-datasets"
DATASET = "dogs-vs-cats"
POSITIVE_LABEL = "dog"


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


def main(args: Namespace) -> int:
    kolena.initialize(verbose=True)

    df_metadata = pd.read_csv(args.dataset_csv, storage_options={"anon": True})
    id_fields = ["locator"]
    ground_truth_field = "label"

    df_serialized_metadata = to_serialized_dataframe(df_metadata[["width", "height"]], column="metadata")
    df_serialized_ground_truth = df_metadata[ground_truth_field].apply(lambda x: to_label_object(x))
    df_datapoints = pd.concat([df_metadata[id_fields], df_serialized_metadata, df_serialized_ground_truth], axis=1)
    register_dataset("dogs-vs-cats", df_datapoints, id_fields)

    return 0


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument(
        "--dataset_csv",
        type=str,
        default=f"s3://{BUCKET}/{DATASET}/meta/metadata.csv",
        help="CSV file with a list of image `locator` and its `label`. See default CSV for details",
    )
    sys.exit(main(ap.parse_args()))
