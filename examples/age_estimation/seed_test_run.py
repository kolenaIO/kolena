import os
import sys
from argparse import ArgumentParser
from argparse import Namespace
from typing import List

import pandas as pd
from kolena_contrib.age_estimation.evaluator import AgeEstimationEvaluator
from kolena_contrib.age_estimation.workflow import Inference
from kolena_contrib.age_estimation.workflow import Model
from kolena_contrib.age_estimation.workflow import TestSample
from kolena_contrib.age_estimation.workflow import TestSuite

import kolena
from kolena.workflow import test

BUCKET = "kolena-public-datasets"
DATASET = "labeled-faces-in-the-wild"
LOCAL_DIR = "/data/open-source/lfw"


def local_path_as_locator(local_path: str) -> str:
    relative_path = local_path.split(LOCAL_DIR)[1].strip("/")
    return f"s3://{BUCKET}/{DATASET}/{relative_path}"


def seed_test_run(model_name: str, test_suite_names: List[str]) -> None:
    df_results = pd.read_csv(f"s3://{BUCKET}/{DATASET}/predictions/{model_name}.csv")
    if LOCAL_DIR in df_results.iloc[0]["image_path"]:
        df_results["image_path"] = df_results["image_path"].apply(lambda x: local_path_as_locator(x))

    def infer(test_sample: TestSample) -> Inference:
        age = df_results[df_results["image_path"] == test_sample.locator]["age"].values[0]
        return Inference(age=age if age != -1 else None)

    evaluator = AgeEstimationEvaluator()
    print(f"Evaluator: {evaluator}")

    model = Model(f"{model_name} [age estimation]", infer=infer)
    print(f"Model: {model}")

    for test_suite_name in test_suite_names:
        test_suite = TestSuite.load(test_suite_name)
        print(f"Test Suite: {test_suite}")

        test(model, test_suite, evaluator)


def main(args: Namespace) -> int:
    env_token = "KOLENA_TOKEN"
    print(f"initializing with environment variables ${env_token}")
    kolena.initialize(os.environ[env_token], verbose=True)

    seed_test_run(args.model_name, args.test_suite_names)
    return 0


if __name__ == "__main__":
    ap = ArgumentParser("kolena_contrib.age_estimation")
    ap.add_argument("model_name", help="Name of model in directory to test")
    ap.add_argument("test_suite_names", nargs="+", help="Name of test suite(s) to test.")

    sys.exit(main(ap.parse_args()))
