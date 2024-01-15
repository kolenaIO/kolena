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
import sys
from argparse import ArgumentParser
from argparse import Namespace

from semantic_segmentation.activation_map_uploader import ActivationMapUploader
from semantic_segmentation.constants import BUCKET
from semantic_segmentation.constants import DATASET
from semantic_segmentation.constants import MODEL_NAME

import kolena
from kolena.dataset import download_dataset


def upload_activation_map(model_name: str, write_bucket: str) -> None:
    inf_locator_prefix = f"s3://{BUCKET}/{DATASET}/results/{model_name}/"
    map_locator_prefix = f"s3://{write_bucket}/{DATASET}/inferences/{model_name}/activation/"
    uploader = ActivationMapUploader(inf_locator_prefix, map_locator_prefix)
    df_dataset = download_dataset(DATASET).head(5)
    uploader.submit(df_dataset["basename"].tolist())


def main(args: Namespace) -> int:
    kolena.initialize(verbose=True)
    upload_activation_map(MODEL_NAME[args.model], args.write_bucket)
    return 0


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument(
        "model",
        type=str,
        default="pspnet_r101",
        nargs="?",
        choices=list(MODEL_NAME.keys()),
        help="Name of the model to test.",
    )
    ap.add_argument(
        "--write-bucket",
        type=str,
        required=True,
        help="Name of AWS S3 bucket with write access to upload result masks to.",
    )

    sys.exit(main(ap.parse_args()))
