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
from typing import Dict

import pandas as pd
from speaker_diarization.evaluator import evaluate_speaker_diarization
from speaker_diarization.utils import realign_labels
from speaker_diarization.workflow import Inference
from speaker_diarization.workflow import Model
from speaker_diarization.workflow import TestSample
from speaker_diarization.workflow import TestSuite

import kolena
from kolena.workflow.annotation import LabeledTimeSegment
from kolena.workflow.test_run import test

BUCKET = "kolena-public-datasets"
DATASET = "ICSI-corpus"

TEST_SUITE_NAMES = [
    f"{DATASET} :: average amplitude",
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
    align_speakers: bool,
) -> None:
    def infer(sample: TestSample) -> Inference:
        inference_path = sample.metadata["transcription_path"].replace("audio/", f"{mod}_inferences/")
        inference_df = pd.read_csv(f"s3://{BUCKET}/{DATASET}/{inference_path}/", storage_options={"anon": True})
        if align_speakers:
            gt_df = pd.read_csv(
                f"s3://{BUCKET}/{DATASET}/{sample.metadata['transcription_path'][:-4] + '_cleaned.csv'}",
                storage_options={"anon": True},
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
                for _, row in inference_df.iterrows()
            ],
        )

    print(f"working on {mod} and {test_suite.name} v{test_suite.version}")
    model = Model(f"{mod}", infer=infer, metadata={**MODEL_METADATA[mod]})

    test(model, test_suite, evaluate_speaker_diarization, reset=True)


def main(args: Namespace) -> None:
    kolena.initialize(verbose=True)

    mod = "gcp-stt-video"

    print("loading test suite")
    test_suites = TestSuite.load_all(tags={DATASET})
    for test_suite in test_suites:
        seed_test_run(mod, test_suite, args.align_speakers)


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument(
        "--align-speakers",
        type=bool,
        default=False,
        help="Specify whether to perform speaker alignment between the GT and Inf in the preprocessing step.",
    )
    main(ap.parse_args())
