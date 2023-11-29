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
from collections import OrderedDict
from typing import Optional

from rag_qa.constants import HALU_DIALOG
from rag_qa.constants import HALU_QA
from rag_qa.constants import HALU_SUMMARIZATION
from rag_qa.constants import SQUAD2_DEV
from rag_qa.constants import SQUAD2_TRAIN
from rag_qa.data_loader import load_halu_dialog
from rag_qa.data_loader import load_halu_qa
from rag_qa.data_loader import load_halu_summarization
from rag_qa.data_loader import load_squad2_dev
from rag_qa.data_loader import load_squad2_train
from rag_qa.workflow import GroundTruth
from rag_qa.workflow import TestCase
from rag_qa.workflow import TestSuite

import kolena
from kolena.workflow import Text


def clean_name(name: str) -> str:
    return name[9:] if name.startswith("metadata_") else name


data_loader = OrderedDict(
    [
        (SQUAD2_DEV, load_squad2_dev),
        (SQUAD2_TRAIN, load_squad2_train),
        (HALU_QA, load_halu_qa),
        (HALU_DIALOG, load_halu_dialog),
        (HALU_SUMMARIZATION, load_halu_summarization),
    ],
)

df_gt_columns = {
    SQUAD2_DEV: ["answers", "is_impossible", "plausible_answers"],
    SQUAD2_TRAIN: ["answers", "is_impossible", "plausible_answers"],
    HALU_QA: ["right_answer", "hallucinated_answer"],
    HALU_DIALOG: ["right_response", "hallucinated_response"],
    HALU_SUMMARIZATION: ["right_summary", "hallucinated_summary"],
}


def seed_benchmark(benchmark: str, test_suite: Optional[str] = None) -> None:
    df = data_loader[benchmark]()
    gt_columns = df_gt_columns[benchmark]
    ts_columns = [col for col in df.columns if col not in gt_columns]
    test_samples = df[ts_columns].to_dict(orient="records")
    ground_truths = df[gt_columns].to_dict(orient="records")
    test_suite_name = test_suite or benchmark

    # Create a list of every test sample with its ground truth
    test_samples_and_ground_truths = [(Text(**ts), GroundTruth(**gt)) for ts, gt in zip(test_samples, ground_truths)]
    complete_test_case = TestCase(
        f"complete {test_suite_name}",
        description=f"complete {test_suite_name}",
        test_samples=test_samples_and_ground_truths,
        reset=True,
    )
    test_suite = TestSuite(test_suite_name, test_cases=[complete_test_case], reset=True)
    print(f"created test suite: {test_suite.name} v{test_suite.version}")


def main(args: Namespace) -> None:
    kolena.initialize(verbose=True)

    benchmark = args.benchmark
    if benchmark:
        seed_benchmark(benchmark, args.test_suite)
    else:
        # seed all datasets
        for dataset in data_loader.keys():
            seed_benchmark(dataset)


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument(
        "--benchmark",
        choices=[SQUAD2_DEV, SQUAD2_TRAIN, HALU_QA, HALU_DIALOG, HALU_SUMMARIZATION],
        help="Name of the benchmark to seed.",
    )
    ap.add_argument("--test-suite", help="test suite name to create")
    main(ap.parse_args())
