# Copyright 2021-2023 Kolena Inc.
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
from typing import Callable
from typing import Dict

import pandas as pd
from evaluator import evaluate_audio_recognition
from utils import realign_labels
from workflow import Inference
from workflow import Model
from workflow import TestSample
from workflow import TestSuite

import kolena
from kolena.workflow.annotation import LabeledTimeSegment
from kolena.workflow.test_run import test

BUCKET = "kolena-public-datasets"
DATASET = "ICSI-corpus"

TEST_SUITE_NAMES = [
    f"{DATASET} :: average amplitude",
    f"{DATASET} :: zero crossing rate",
    f"{DATASET} :: energy",
]

MODEL_A = {
    "model family": "GCP Speech To Text",
    "model name": "gcp-stt-video",
    "description": "Speaker diarization model optimized for videos.",
}

MODEL_METADATA: Dict[str, Dict[str, str]] = {
    "gcp-stt-video": MODEL_A,
}


def seed_test_run(
    mod: str,
    test_suite: TestSuite,
    infs: pd.DataFrame,
    align_speakers: bool,
) -> None:
    def get_stored_inferences(
        df: pd.DataFrame,
    ) -> Callable[[TestSample], Inference]:
        def infer(sample: TestSample) -> Inference:
            inference_path = sample.metadata["transcription_path"].replace("audio/", f"{mod}_inferences/")
            inference_df = pd.read_csv(f"s3://{BUCKET}/{DATASET}/{inference_path}/")
            if align_speakers:
                gt_df = pd.read_csv(
                    f"s3://{BUCKET}/{DATASET}/{sample.metadata['transcription_path'][:-4] + '_cleaned.csv'}",
                )
                realign_labels(gt_df, inference_df)

            return Inference(
                transcription=[
                    LabeledTimeSegment(
                        start=row.starttime,
                        end=row.endtime,
                        label=row.text,
                        group=row.speaker,
                    )
                    for idx, row in inference_df.iterrows()
                ],
            )

        return infer

    model_name = mod
    print(f"working on {model_name} and {test_suite.name} v{test_suite.version}")
    infer = get_stored_inferences(infs)
    model = Model(f"{mod}", infer=infer, metadata={**MODEL_METADATA[model_name]})

    test(model, test_suite, evaluate_audio_recognition, reset=True)


def main(args: Namespace) -> None:
    kolena.initialize(verbose=True)

    mod = args.model
    print("loading inference CSV")
    csv_to_use = f"s3://{BUCKET}/{DATASET}/metadata.csv"
    columns_of_interest = [
        "audio_path",
    ]
    df_results = pd.read_csv(csv_to_use, usecols=columns_of_interest)

    if args.test_suite is None:
        print("loading test suite")
        test_suites = [TestSuite.load(name) for name in TEST_SUITE_NAMES]
        for test_suite in test_suites:
            seed_test_run(mod, test_suite, df_results, args.align_speakers)
    else:
        test_suite = TestSuite.load(args.test_suite)
        seed_test_run(mod, test_suite, df_results, args.align_speakers)


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument(
        "model",
        type=str,
        choices=sorted(MODEL_METADATA.keys()),
        help="The name of the model to test.",
    )
    ap.add_argument(
        "--test-suite",
        type=str,
        help="Optionally specify a test suite to test. Test against all available test suites when unspecified.",
    )
    ap.add_argument(
        "--align-speakers",
        type=bool,
        default=False,
        help="Specify whether to perform speaker alignment between the GT and Inf in the preprocessing step.",
    )
    main(ap.parse_args())
