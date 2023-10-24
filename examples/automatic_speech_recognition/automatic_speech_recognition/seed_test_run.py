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
from workflow import Inference
from workflow import Model
from workflow import TestSample
from workflow import TestSuite

import kolena
from kolena.workflow.annotation import ClassificationLabel
from kolena.workflow.test_run import test

BUCKET = "kolena-public-datasets"
DATASET = "LibriSpeech"

TEST_SUITE_NAMES = [
    f"{DATASET} :: audio duration",
    f"{DATASET} :: speaker sex",
    f"{DATASET} :: tempo (words per second)",
]

MODEL_A = {
    "model family": "whisper",
    "model name": "whisper-1-default",
    "description": "OpenAI's premier speech-to-text model.",
}

MODEL_B = {
    "model family": "whisper",
    "model name": "whisper-1-translate",
    "description": "OpenAI's premier speech-to-text model w/ English translations.",
}

MODEL_C = {
    "model family": "wav2vec2",
    "model name": "wav2vec2-base-960h",
    "description": "Facebook's Wav2Vec2 model trained on 960h of LibriSpeech audio samples.",
}

MODEL_METADATA: Dict[str, Dict[str, str]] = {
    "whisper-1-default": MODEL_A,
    "whisper-1-translate": MODEL_B,
    "wav2vec2-base-960h": MODEL_C,
}


def seed_test_run(
    mod: str,
    test_suite: TestSuite,
    infs: pd.DataFrame,
) -> None:
    def get_stored_inferences(
        df: pd.DataFrame,
    ) -> Callable[[TestSample], Inference]:
        def infer(sample: TestSample) -> Inference:
            result = df.loc[df["id"] == sample.metadata["file_id"]]
            transcription = result[f"inference_{mod}"].values[0]
            return Inference(
                transcription=ClassificationLabel(transcription),
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
    csv_to_use = f"s3://{BUCKET}/{DATASET}/inference.csv"
    columns_of_interest = [
        "id",
        "locator",
        "text",
        "inference_whisper-1-default",
        "inference_whisper-1-translate",
        "inference_wav2vec2-base-960h",
    ]
    df_results = pd.read_csv(csv_to_use, usecols=columns_of_interest)

    if args.test_suite is None:
        print("loading test suite")
        test_suites = [TestSuite.load(name) for name in TEST_SUITE_NAMES]
        for test_suite in test_suites:
            seed_test_run(mod, test_suite, df_results)
    else:
        test_suite = TestSuite.load(args.test_suite)
        seed_test_run(mod, test_suite, df_results)


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
    main(ap.parse_args())
