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
from pydantic.dataclasses import dataclass
from video_retrieval_by_text.constants import DEFAULT_DATASET_NAME
from video_retrieval_by_text.constants import MODELS
from video_retrieval_by_text.constants import RESULTS

import kolena
from kolena._utils.validators import ValidatorConfig
from kolena.asset import BaseVideoAsset
from kolena.dataset import download_dataset
from kolena.dataset import upload_results


@dataclass(frozen=True, config=ValidatorConfig)
class VideoAssetWithScoreAndRank(BaseVideoAsset):
    similarity: float
    rank: int


def generate_results(similarity_df: pd.DataFrame, video_series: pd.Series) -> pd.DataFrame:
    results = []
    for caption_id, scores in similarity_df.iterrows():
        video_id = _caption_id_to_video_id(caption_id)
        sorted_scores = scores.sort_values(ascending=False)
        rank = sorted_scores.index.get_loc(video_id) + 1
        results.append(
            dict(
                caption_id=caption_id,
                similarity=sorted_scores[video_id],
                rank=rank,
                reciprocal_rank=1 / rank,
                total_miss=rank > 50,
                **{f"is_top_{k:02d}": rank <= k for k in [1, 5, 10, 25, 50]},
                top_10=[
                    VideoAssetWithScoreAndRank(
                        **video_series[video_id]._to_dict(),
                        similarity=similarity,
                        rank=rank,
                    )
                    for rank, (video_id, similarity) in enumerate(sorted_scores[:10].items(), start=1)
                ],
            ),
        )

    return pd.DataFrame(results)


def _caption_id_to_video_id(caption_id: str) -> str:
    return caption_id.rsplit("-", maxsplit=1)[0]


def _generate_video_series(dataset_df: pd.DataFrame) -> pd.Series:
    dataset_df["video_id"] = dataset_df["caption_id"].apply(_caption_id_to_video_id)
    dataset_df.drop_duplicates(subset="video_id", inplace=True)
    video_df = dataset_df.set_index("video_id")
    return video_df["video"]


def run(args: Namespace) -> None:
    kolena.initialize(verbose=True)
    video_series = _generate_video_series(download_dataset(args.dataset))
    for model in args.models:
        similarity_df = pd.read_csv(f"{RESULTS}/{model}/similarity/similarities.csv.gz", index_col=0)
        results_df = generate_results(similarity_df, video_series)
        upload_results(args.dataset, model, results_df)


def main() -> None:
    ap = ArgumentParser()
    ap.add_argument("models", nargs="*", choices=MODELS, type=str, help="Select model(s) to test.")
    ap.add_argument(
        "-d",
        "--dataset",
        default=DEFAULT_DATASET_NAME,
        type=str,
        help="Optionally specify the name of the dataset to test against.",
    )
    run(ap.parse_args())


if __name__ == "__main__":
    main()
