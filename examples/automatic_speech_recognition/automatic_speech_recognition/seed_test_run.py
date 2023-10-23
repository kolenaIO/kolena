from argparse import ArgumentParser
from argparse import Namespace
from typing import Callable
from typing import Dict
from typing import Tuple

import pandas as pd
from evaluator import evaluate_audio_recognition
from workflow import Inference
from workflow import Model
from workflow import TestSample
from workflow import TestSuite

import kolena
from kolena.workflow.test_run import test
from kolena.workflow.annotation import ClassificationLabel

BUCKET = 'kolena-public-datasets'
DATASET = 'LibriSpeech'

MODEL_MAP: Dict[str, Tuple[str, str]] = {
    "whisper-1-default": ("whisper-1-default", f"whisper-1-default"),
    "whisper-1-translate": ("whisper-1-translate", f"whisper-1-translate"),
    "wav2vec2-base-960h": ("wav2vec2-base-960h", f"wav2vec2-base-960h"),
}

TEST_SUITE_NAMES = [
    # f"{DATASET} :: word count",
    f"{DATASET} :: audio duration",
    # f"{DATASET} :: speaker sex",
    # f"{DATASET} :: longest word length",
    f"{DATASET} :: max pitch",
    # f"{DATASET} :: energy",
    # f"{DATASET} :: zero crossing rate",
    f"{DATASET} :: tempo (words per second)",
]

COMMON_METADATA = {
    "id": "1272-128104-0000",
    "locator": "./dev-clean\1272\128104\1272-128104-0000.flac",
    "text": "mister quilter is the apostle of the middle classes and we are glad to welcome his gospel",
    "speaker_sex": ' M ',
    "duration_seconds": 5.855,
    "max_pitch": 107.211,
    "energy": 0.00381,
    "word_count": 17,
    "longest_word_len": 7,
    "zero_crossing_rate": 2876.510,
}

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
    mod: Tuple[str, str],
    test_suite: TestSuite,
    infs: pd.DataFrame,
) -> None:
    def get_stored_inferences(
        df: pd.DataFrame,
    ) -> Callable[[TestSample], Inference]:
        def infer(sample: TestSample) -> Inference:
            result = df.loc[df["id"] == sample.metadata['file_id']]
            transcription = result[f"inference_{mod[0]}"].values[0]
            return Inference(
                transcription=ClassificationLabel(transcription)
            )

        return infer

    model_name = mod[0]
    print(f"working on {model_name} and {test_suite.name} v{test_suite.version}")
    infer = get_stored_inferences(infs)
    model = Model(f"{mod[1]}", infer=infer, metadata={**MODEL_METADATA[model_name], **COMMON_METADATA})
    print(model)

    test(model, test_suite, evaluate_audio_recognition, reset=True)


def main(args: Namespace) -> None:
    kolena.initialize(verbose=True)

    mod = MODEL_MAP[args.model]
    print("loading inference CSV")
    s3_path = f"s3://{BUCKET}/{DATASET}/inference.csv"
    csv_to_use = s3_path if args.local_csv is None else args.local_csv
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
            print(test_suite.name)
            seed_test_run(mod, test_suite, df_results)
    else:
        test_suite = TestSuite.load(args.test_suite)
        print(test_suite.name)
        seed_test_run(mod, test_suite, df_results)


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument(
        "model",
        type=str,
        choices=sorted(MODEL_MAP.keys()),
        help="The name of the model to test.")
    ap.add_argument(
        "--test-suite",
        type=str,
        help="Optionally specify a test suite to test. Test against all available test suites when unspecified.",
    )
    ap.add_argument(
        "--local-csv",
        type=str,
        help="Optionally specify a local results CSV to use. Defaults to CSVs stored in S3 when absent.",
    )
    main(ap.parse_args())
