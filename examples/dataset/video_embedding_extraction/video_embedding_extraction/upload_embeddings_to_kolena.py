# Copyright 2021-2025 Kolena Inc.
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
import pickle
from typing import Dict

import numpy as np
import pandas as pd
from tqdm import tqdm

from kolena.dataset import upload_dataset_embeddings

S3_LOCATOR_PREFIX = "s3://kolena-public-datasets/JAAD/JAAD_clips/"


def load_embeddings(pickle_path: str) -> Dict[str, np.ndarray]:
    """
    Load embeddings from pickle file.

    Args:
        pickle_path (str): Path to the pickle file containing embeddings

    Returns:
        dict: Dictionary mapping video filenames to their embeddings
    """
    with open(pickle_path, "rb") as f:
        embeddings = pickle.load(f)
    return embeddings


def create_embeddings_dataframe(
    embeddings: Dict[str, np.ndarray],
    video_dir: str,
    s3_prefix: str = S3_LOCATOR_PREFIX,
) -> pd.DataFrame:
    """
    Create a DataFrame with video locators and their embeddings.

    Args:
        embeddings (dict): Dictionary mapping video filenames to their embeddings
        video_dir (str): Directory containing the video files
        s3_prefix (str): S3 prefix to use for locators

    Returns:
        pd.DataFrame: DataFrame with locator and embedding columns
    """
    records = []

    for video_name, embedding in tqdm(embeddings.items(), desc="Creating embedding records"):
        if video_name not in os.listdir(video_dir):
            print(f"Warning: Video file {video_name} not found in {video_dir}")
            continue

        if isinstance(embedding, np.ndarray):
            embedding = embedding.squeeze()  # Remove any extra dimensions

        s3_locator = os.path.join(s3_prefix, video_name)

        records.append(
            {
                "locator": s3_locator,
                "embedding": embedding,
            },
        )

    return pd.DataFrame.from_records(records)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload video embeddings to Kolena dataset",
    )
    parser.add_argument(
        "--embeddings_file",
        type=str,
        required=True,
        help="Path to the pickle file containing video embeddings",
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        required=True,
        help="Directory containing the video files (for verification)",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Name of the existing Kolena dataset to upload embeddings to",
    )
    parser.add_argument(
        "--embedding_key",
        type=str,
        required=True,
        help="Unique identifier for these embeddings (e.g., 'viclip-video-embeddings')",
    )
    parser.add_argument(
        "--s3_prefix",
        type=str,
        default=S3_LOCATOR_PREFIX,
        help=f"S3 prefix for video locators (default: {S3_LOCATOR_PREFIX})",
    )

    args = parser.parse_args()

    if not os.path.exists(args.embeddings_file):
        raise FileNotFoundError(f"Embeddings file not found: {args.embeddings_file}")

    if not os.path.exists(args.video_dir):
        raise FileNotFoundError(f"Video directory not found: {args.video_dir}")

    print(f"Loading embeddings from {args.embeddings_file}")
    embeddings = load_embeddings(args.embeddings_file)

    print("Creating embeddings DataFrame...")
    df_embeddings = create_embeddings_dataframe(embeddings, args.video_dir, args.s3_prefix)

    if len(df_embeddings) == 0:
        print("No valid embeddings found. Please check your video directory and embeddings file.")
        return

    print(
        f"Uploading {len(df_embeddings)} embeddings to dataset "
        f"'{args.dataset_name}' with key '{args.embedding_key}'",
    )
    try:
        upload_dataset_embeddings(
            dataset_name=args.dataset_name,
            key=args.embedding_key,
            df_embedding=df_embeddings,
        )
        print("Embeddings uploaded successfully!")
    except Exception as e:
        print(f"Error uploading embeddings: {str(e)}")
        print("Make sure the dataset exists and you have the correct permissions.")
        print("Also ensure that KOLENA_TOKEN environment variable is set correctly.")


if __name__ == "__main__":
    main()
