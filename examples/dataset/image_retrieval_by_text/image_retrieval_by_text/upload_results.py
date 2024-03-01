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
import json
import os
from argparse import ArgumentParser
from argparse import Namespace
from dataclasses import dataclass
from os import path
from typing import List
from typing import Union

import boto3
import numpy as np
import pandas as pd
import torch
from image_retrieval_by_text.constants import BUCKET
from image_retrieval_by_text.constants import DATASET
from image_retrieval_by_text.constants import EMBEDDING_STORAGE_DIR
from image_retrieval_by_text.constants import MODELS
from PIL import Image
from tqdm import tqdm
from transformers import AlignModel
from transformers import AltCLIPModel
from transformers import AltCLIPProcessor
from transformers import AutoModel
from transformers import AutoProcessor
from transformers import CLIPModel
from transformers import CLIPProcessor

import kolena
from kolena.asset import ImageAsset
from kolena.dataset import download_dataset
from kolena.dataset import upload_results


@dataclass(frozen=True)
class ImageAssetWithScoreAndRank(ImageAsset):
    similarity: float
    rank: int


def _transform_top_10(top_ten: list[dict]) -> List[ImageAssetWithScoreAndRank]:
    current_rank = 1
    new_top_ten = []
    for item in top_ten:
        new_top_ten.append(
            ImageAssetWithScoreAndRank(
                similarity=item["similarity"],
                locator=item["locator"],
                rank=current_rank,
            ),
        )
        current_rank += 1
    return new_top_ten


def _transform_data(df_raw_csv: pd.DataFrame) -> pd.DataFrame:
    df_raw_csv["top_10"] = df_raw_csv["top_10"].apply(
        lambda x: _transform_top_10(json.loads(x)),
    )
    return df_raw_csv


def _get_image_from_s3(image_s3_url: str) -> Image:
    """
    Download an image from an S3 bucket.

    :param bucket_name: Name of the S3 bucket.
    :param object_key: Key of the object in the S3 bucket.
    :return: An image object.
    """
    bucket_name, *parts = image_s3_url[5:].split("/")
    file_stream = boto3.resource("s3").Bucket(bucket_name).Object("/".join(parts)).get()["Body"]
    return Image.open(file_stream)


def _get_model_and_processor(
    model_name: str,
) -> tuple[
    Union[AlignModel, AutoModel, AltCLIPModel, CLIPModel],
    Union[AltCLIPProcessor, AutoProcessor, CLIPProcessor],
]:
    if model_name == "kakaobrain_align-base":
        model_name_full = "kakaobrain/align-base"
        model = AlignModel.from_pretrained(model_name_full)
        processor = AutoProcessor.from_pretrained(model_name_full)
    elif model_name == "openai_clip-vit-base-patch32":
        model_name_full = "openai/clip-vit-base-patch32"
        model = CLIPModel.from_pretrained(model_name_full)
        processor = CLIPProcessor.from_pretrained(model_name_full)
    elif model_name == "BAAI_AltCLIP":
        model_name_full = "BAAI/AltCLIP"
        model = AltCLIPModel.from_pretrained(model_name_full)
        processor = AltCLIPProcessor.from_pretrained(model_name_full)
    elif model_name == "google_siglip-base-patch16-224":
        model_name_full = "google/siglip-base-patch16-224"
        model = AutoModel.from_pretrained(model_name_full)
        processor = AutoProcessor.from_pretrained(model_name_full)
    else:
        raise ValueError(f"Model {model_name} not found.")
    return model, processor


def _extract_embeddings(
    model: Union[AlignModel, AutoModel, AltCLIPModel, CLIPModel],
    processor: Union[AltCLIPProcessor, AutoProcessor, CLIPProcessor],
    dataset: pd.DataFrame,
    local_image_dir: str,
    model_name: str,
    save_dir: str = "",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    image_records = []
    text_records = []
    for image_name, df_batch in tqdm(
        dataset[["image_url", "image_name", "caption", "caption_id"]].groupby("image_name"),
    ):
        if local_image_dir:
            try:
                images = [Image.open(path.join(local_image_dir, image_name))]
            except Exception:
                images = [_get_image_from_s3(df_batch.iloc[0]["image_url"].locator)]
        else:
            images = [_get_image_from_s3(df_batch.iloc[0]["image_url"].locator)]
        texts = [record.caption for record in df_batch.itertuples()]

        try:
            # important: match the padding strategy provided in HF tutorial
            padding = "max_length" if model_name == "google_siglip-base-patch16-224" else True
            inputs = processor(
                text=texts,
                images=images,
                padding=padding,
                return_tensors="pt",
            )
            with torch.no_grad():
                outputs = model(**inputs)
        except Exception as e:
            print(e)
            continue

        image_records.append(dict(image_name=image_name, embedding=outputs.image_embeds.squeeze().numpy()))
        for i in range(outputs.text_embeds.shape[0]):
            text_records.append(
                dict(caption_id=df_batch.iloc[i]["caption_id"], embedding=outputs.text_embeds[i].numpy()),
            )

    df_img_emb = pd.DataFrame(image_records)
    df_txt_emb = pd.DataFrame(text_records)
    if save_dir:
        os.makedirs(f"{save_dir}/{model_name}", exist_ok=True)
        df_img_emb.to_parquet(f"{save_dir}/{model_name}/img.parquet", index=False)
        df_txt_emb.to_parquet(f"{save_dir}/{model_name}/txt.parquet", index=False)
    return df_img_emb, df_txt_emb


def retrieve_images_by_text(dataset: pd.DataFrame, df_img_emb: pd.DataFrame, df_txt_emb: pd.DataFrame) -> pd.DataFrame:
    """
    Get the 10 most similar images for each text in the dataset according to embedding similarity.
    :param dataset: The dataset.
    :param df_img_emb: The image embeddings.
    :param df_txt_emb: The text embeddings.
    """

    image_embeddings = np.concatenate(df_img_emb["embedding"].apply(lambda a: a.reshape((1, a.shape[0]))), axis=0)

    image_index_by_image_name = {r.image_name: i for i, r in enumerate(df_img_emb.itertuples())}
    image_name_by_caption_id = {r.caption_id: r.image_name for r in dataset.itertuples()}
    locator_by_image_index = {
        image_index_by_image_name[r.image_name]: r.image_url.locator
        for r in dataset.itertuples()
        if r.image_name in image_index_by_image_name.keys()
    }

    results = []
    for text_index, record in tqdm(enumerate(df_txt_emb.itertuples()), total=len(df_txt_emb)):
        text_emb = record.embedding.reshape((1, record.embedding.shape[0]))
        sim_arr = (text_emb @ image_embeddings.T).squeeze()
        order = sim_arr.argsort()[::-1]
        ranks = order.argsort()
        image_index = image_index_by_image_name[image_name_by_caption_id[record.caption_id]]
        rank = ranks[image_index] + 1
        results.append(
            dict(
                caption_id=record.caption_id,
                similarity=sim_arr[image_index],
                rank=rank,
                reciprocal_rank=1 / rank,
                total_miss=rank > 50,
                **{f"is_top_{k:02d}": rank <= k for k in [1, 5, 10, 25, 50]},
                top_10=[
                    ImageAssetWithScoreAndRank(
                        locator=locator_by_image_index[index],
                        similarity=sim_arr[index],
                        rank=r + 1,
                    )
                    for r, index in enumerate(order[:10])
                ],
            ),
        )

    return pd.DataFrame(results)


def run(args: Namespace) -> None:
    kolena.initialize(verbose=True)
    if not args.run_inference:
        pred_df_csv = pd.read_csv(
            f"s3://{BUCKET}/coco-2014-val/coco-2014-val_image-retrieval-by-text/results/raw/{args.model}-raw.csv",
            storage_options={"anon": True},
        )
        pred_df = _transform_data(pred_df_csv)
    else:
        print("downloading dataset")
        dataset = download_dataset(args.dataset)

        if os.path.exists(f"{EMBEDDING_STORAGE_DIR}/{args.model}/img.parquet") and os.path.exists(
            f"{EMBEDDING_STORAGE_DIR}/{args.model}/txt.parquet",
        ):
            print(f"loading pre-computed embeddings stored at {EMBEDDING_STORAGE_DIR}")
            df_img_emb = pd.read_parquet(f"{EMBEDDING_STORAGE_DIR}/{args.model}/img.parquet")
            df_txt_emb = pd.read_parquet(f"{EMBEDDING_STORAGE_DIR}/{args.model}/txt.parquet")

        else:
            print("loading model and processor")
            model, processor = _get_model_and_processor(args.model)

            print("extracting image and text embeddings")
            df_img_emb, df_txt_emb = _extract_embeddings(
                model,
                processor,
                dataset,
                args.local_image_dir,
                args.model,
                save_dir="embeddings",
            )

        print("calculating similarity between text embeddings and image embeddings")
        pred_df = retrieve_images_by_text(dataset, df_img_emb, df_txt_emb)

    upload_results(args.dataset, args.model, pred_df)


def main() -> None:
    ap = ArgumentParser()
    ap.add_argument(
        "--model",
        type=str,
        choices=MODELS,
        default=MODELS[0],
        help=f"Name of the model to test. If you want to run inference {MODELS[-1]} model is recommended, "
        f"as it runs faster (should complete in 30 minutes)",
    )
    ap.add_argument("--dataset", type=str, default=DATASET, help="Optionally specify a custom dataset name to test.")
    ap.add_argument(
        "--run-inference",
        type=bool,
        default=False,
        help="Optionally specify whether to run inference. If this is False, pre-computed inference results "
        "will be used",
    )
    ap.add_argument(
        "--local-image-dir",
        type=str,
        default="",
        help="Optionally specify a local directory that stores the images to make run inference faster",
    )
    run(ap.parse_args())


if __name__ == "__main__":
    main()
