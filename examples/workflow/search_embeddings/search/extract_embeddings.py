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
import argparse
import os
from typing import Iterator
from typing import List
from typing import Optional
from typing import Tuple

import boto3
from kembed.util import extract_and_upload_embeddings
from PIL import Image


BUCKET = "kolena-public-datasets"
DATASET = "labeled-faces-in-the-wild"


def image_locators_from_s3_path(s3_path: str, local_dir: Optional[str] = None) -> List[Tuple[str, Optional[str]]]:
    print(f"The input s3 path is: {s3_path}")
    if not s3_path.startswith("s3://"):
        raise ValueError(f"invalid input path: {s3_path}")

    bucket_name, *parts = s3_path[5:].split("/")
    prefix = "/".join(parts)
    bucket = boto3.resource("s3").Bucket(bucket_name)

    locators_and_accessors: List[Tuple[str, Optional[str]]] = []
    objects = bucket.objects.filter(Prefix=prefix)

    for obj in objects:
        locator = "s3://" + bucket_name + "/" + obj.key
        if local_dir is None:
            locators_and_accessors.append((locator, None))
        else:
            target = os.path.join(local_dir, os.path.relpath(obj.key, prefix))
            if not os.path.exists(target):
                raise ValueError(f"missing local file: {target}")
            locators_and_accessors.append((locator, target))

    return locators_and_accessors


def load_image_from_accessor(accessor: str) -> Image:
    if accessor.startswith("s3://"):
        bucket_name, *parts = accessor[5:].split("/")
        file_stream = boto3.resource("s3").Bucket(bucket_name).Object("/".join(parts)).get()["Body"]
        return Image.open(file_stream)
    else:  # local path
        return Image.open(accessor)


def iter_image_paths(image_accessors: List[Tuple[str, Optional[str]]]) -> Iterator[Tuple[str, Image.Image]]:
    for locator, filepath in image_accessors:
        image = load_image_from_accessor(filepath) if filepath is not None else load_image_from_accessor(locator)
        yield locator, image


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    s3_path = f"s3://{BUCKET}/{DATASET}/imgs/"
    parser.add_argument(
        "--local-path",
        type=str,
        help=f"Local path where files have already been pre-downloaded (to the same relative path as {s3_path})",
    )

    args = parser.parse_args()
    local_path: str = args.local_path
    if local_path is None or local_path == "":
        print(
            "local-path argument is unset. Please note that pre-downloading all images in batch and later "
            "extracting embeddings with the local-path flag will be significantly faster than streaming the "
            "extraction.",
        )

    locators_and_filepaths = image_locators_from_s3_path(s3_path, local_path)
    if len(locators_and_filepaths) == 0:
        raise ValueError(f"invalid input path: {s3_path}")

    extract_and_upload_embeddings(iter_image_paths(locators_and_filepaths))
