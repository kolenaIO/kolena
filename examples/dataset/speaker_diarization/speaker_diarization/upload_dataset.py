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

from speaker_diarization.data_loader import DATASET
from speaker_diarization.data_loader import load_data

import kolena
from kolena.dataset import register_dataset


def main(args: Namespace) -> None:
    kolena.initialize(verbose=True)
    df = load_data()
    sample_count = args.sample_count
    if sample_count:
        df = df[:sample_count]

    register_dataset(args.dataset_name, df)


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument("--dataset-name", type=str, default=DATASET, help="Name of the dataset")
    ap.add_argument(
        "--sample-count",
        type=int,
        default=0,
        help="Number of samples to use, all samples are used if 0",
    )

    main(ap.parse_args())
