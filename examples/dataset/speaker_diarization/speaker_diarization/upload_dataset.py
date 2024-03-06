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

from kolena.dataset import upload_dataset


def run(args: Namespace) -> None:
    df = load_data()
    sample_count = args.sample_count
    if sample_count:
        df = df[:sample_count]

    upload_dataset(args.dataset, df)


def main() -> None:
    ap = ArgumentParser()
    ap.add_argument("--dataset", type=str, default=DATASET, help="Optionally specify a dataset name to upload.")
    ap.add_argument(
        "--sample-count",
        type=int,
        default=0,
        help="Number of samples to use. All samples are used by default.",
    )
    run(ap.parse_args())


if __name__ == "__main__":
    main()
