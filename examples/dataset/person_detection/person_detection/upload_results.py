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
from argparse import ArgumentParser
from argparse import Namespace

import pandas as pd
from person_detection.constants import BUCKET
from person_detection.constants import DATASET
from person_detection.constants import MODELS

import kolena.io
from kolena._experimental.object_detection import upload_object_detection_results
from kolena.annotation import ScoredLabeledBoundingBox


def load_data(df_pred_csv: pd.DataFrame) -> pd.DataFrame:
    df_pred_csv["raw_inferences"] = df_pred_csv["raw_inferences"].apply(
        lambda bboxes: [ScoredLabeledBoundingBox(**bbox.__dict__) for bbox in bboxes],
    )
    return df_pred_csv


def run(args: Namespace) -> None:
    kolena.initialize(verbose=True)

    df_pred_csv = kolena.io.dataframe_from_csv(
        f"s3://{BUCKET}/{DATASET}/results/raw/{args.model}.csv",
        storage_options={"anon": True},
    )
    df_pred = load_data(df_pred_csv)

    upload_object_detection_results(args.dataset, args.model, df_pred)


def main() -> None:
    ap = ArgumentParser()
    ap.add_argument("model", type=str, choices=MODELS, help="Name of the model to test.")
    ap.add_argument("--dataset", type=str, default=DATASET, help="Optionally specify a custom dataset name to test.")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
