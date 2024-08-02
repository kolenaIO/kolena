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
from named_entity_recognition.constants import DATASET
from named_entity_recognition.metrics import evaluate
from named_entity_recognition.tag_map import TagMap
from tqdm import tqdm
from transformers import pipeline

from kolena.annotation import LabeledTextSegment
from kolena.dataset import download_dataset
from kolena.dataset import upload_results


def run(args: Namespace) -> None:
    model_name = "obi/deid_roberta_i2b2"

    df_dataset = download_dataset(args.dataset)

    pipe = pipeline("token-classification", model=model_name, aggregation_strategy="max")

    predictions = []
    results = []
    for record in tqdm(df_dataset.itertuples(), total=len(df_dataset)):
        entities = []
        for entity in pipe(record.text):
            if entity["start"] >= entity["end"]:
                continue

            entities.append(
                LabeledTextSegment(
                    text_field="text",
                    label=entity["entity_group"],
                    start=entity["start"],
                    end=entity["end"],
                    score=entity["score"],  # type: ignore[call-arg]
                ),
            )
        predictions.append(
            dict(
                document_id=record.document_id,
                phi=entities,
            ),
        )
        # Coerce into LabeledTextSegment for comparison in `evaluate`
        ground_truths = [LabeledTextSegment(**segment.__dict__) for segment in record.phi]
        tags = TagMap()._tag_map.keys()
        tp, fp, fn, metrics = evaluate(ground_truths, entities, tags)
        results.append(
            dict(
                TP=tp,
                FP=fp,
                FN=fn,
                counts=metrics,
            ),
        )

    df_results = pd.concat([pd.DataFrame(predictions), pd.DataFrame(results)], axis=1)
    upload_results(args.dataset, model_name, df_results)


def main() -> None:
    ap = ArgumentParser()
    ap.add_argument(
        "--dataset",
        type=str,
        default=DATASET,
        help="Optionally specify a custom dataset name to upload.",
    )
    run(ap.parse_args())


if __name__ == "__main__":
    main()
