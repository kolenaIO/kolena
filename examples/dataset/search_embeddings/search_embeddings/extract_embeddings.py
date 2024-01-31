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
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Optional
from typing import Tuple

import boto3
import numpy as np
import pandas as pd
from kembed._clipper import StudioModel
from kembed.util import extract_embeddings
from kembed.util import load_embedding_model
from PIL import Image
from tqdm import tqdm

import kolena
from kolena._experimental.search import upload_dataset_embeddings
from kolena.dataset import download_dataset


BUCKET = "kolena-public-examples"
DATASET = "coco-stuff-10k"
IMAGE_S3_DIR = f"s3://{BUCKET}/{DATASET}/data/images/"
LOCATOR_FIELD = "locator"


def image_locators_from_s3_path(
    s3_locators: List[str],
    local_dir: Optional[str] = None,
) -> List[Tuple[str, Optional[str]]]:
    locators_and_filepaths: List[Tuple[str, Optional[str]]] = []
    for locator in s3_locators:
        if not locator.startswith("s3://"):
            raise ValueError(f"invalid input path: {locator}")

        if local_dir is None:
            locators_and_filepaths.append((locator, None))
        else:
            relative_locator = os.path.relpath(locator[5:], IMAGE_S3_DIR[5:])
            target = os.path.join(local_dir, relative_locator)
            if not os.path.exists(target):
                raise ValueError(f"missing local file: {target}")
            locators_and_filepaths.append((locator, target))

    return locators_and_filepaths


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


def extract_image_embeddings(
    model: StudioModel,
    locators_and_images: Iterable[Tuple[str, Image.Image]],
    batch_size: int = 50,
) -> List[Tuple[str, np.ndarray]]:
    """
    Extract and upload a list of search embeddings corresponding to sample locators.
    Expects to have an exported `KOLENA_TOKEN` environment variable, as per
        [Kolena client documentation](https://docs.kolena.io/installing-kolena/?h=initialize#initialization).

    :param model: Model used to run embedding extraction
    :param locators_and_images: An iterator through PIL Image files and their corresponding locators (as provided to
        the Kolena platform).
    :param batch_size: Batch size for number of images to extract embeddings for simultaneously. Defaults to 50 to
        avoid having too many file handlers open at once.
    """
    locators_and_embeddings = []
    locators: List[str] = []
    images: List[Image.Image] = []

    batch_idx = 0
    for locator, image in tqdm(locators_and_images):
        if batch_idx >= batch_size:
            locators_and_embeddings.extend(list(zip(locators, extract_embeddings(images, model))))
            locators, images = [], []
            batch_idx = 0
        locators.append(locator)
        images.append(image)
        batch_idx += 1

    if len(locators) > 0:
        locators_and_embeddings.extend(list(zip(locators, extract_embeddings(images, model))))
    return locators_and_embeddings


def extract_dataset_embedding(model: StudioModel, df: pd.DataFrame, local_path: Optional[str] = None) -> pd.DataFrame:
    locators_and_filepaths = image_locators_from_s3_path(df[LOCATOR_FIELD].to_list(), local_path)
    locator_and_image_iterator = iter_image_paths(locators_and_filepaths)
    locator_and_embeddings = extract_image_embeddings(model, locator_and_image_iterator)
    return pd.DataFrame(
        {
            LOCATOR_FIELD: [locator for locator, _ in locator_and_embeddings],
            "embedding": [embedding for _, embedding in locator_and_embeddings],
        },
    )


def load_precomputed_embedding() -> pd.DataFrame:
    return pd.read_parquet(f"s3://{BUCKET}/{DATASET}/embeddings/default_model/embeddings.parquet")


def run(run_extraction: bool, dataset_name: str, local_path: str) -> None:
    kolena.initialize(verbose=True)
    df_dataset = download_dataset(dataset_name)

    model, model_key = load_embedding_model()
    if not run_extraction:
        df_embedding = load_precomputed_embedding()
    else:
        df_embedding = extract_dataset_embedding(model, df_dataset, local_path)

    df = df_dataset.merge(df_embedding, on=LOCATOR_FIELD)
    upload_dataset_embeddings(dataset_name, model_key, df)


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--run-extraction",
        type=bool,
        default=False,
        help="Whether to run extraction. A set of pre-extracted embeddings will be used if set to False.",
    )

    ap.add_argument(
        "--dataset-name",
        type=str,
        default=DATASET,
        help="Optionally specify a name of the dataset to upload embeddings",
    )

    ap.add_argument(
        "--local-path",
        type=str,
        default=None,
        help=f"Local path where files have already been pre-downloaded (to the same relative path as {IMAGE_S3_DIR})",
    )

    args = ap.parse_args()
    if args.run_extraction and (args.local_path is None or args.local_path == ""):
        print(
            "local-path argument is unset. Please note that pre-downloading all images in batch and later "
            "extracting embeddings with the local-path flag will be significantly faster than streaming the "
            "extraction.",
        )

    run(args.run_extraction, args.dataset_name, args.local_path)


if __name__ == "__main__":
    main()
