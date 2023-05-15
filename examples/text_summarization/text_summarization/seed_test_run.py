import os
from argparse import ArgumentParser
from argparse import Namespace
from typing import Callable
from typing import Dict
from typing import Tuple

import pandas as pd

import kolena
from .evaluator import evaluate_text_summarization
from .workflow import Inference
from .workflow import Model
from .workflow import TestSample
from .workflow import TestSuite
from kolena.workflow.test_run import test

WORKFLOW = "Text Summarization"
MODEL_MAP: Dict[str, Tuple[str, str]] = {
    "ada": ("text-ada-001", f"ADA (GPT-3, {WORKFLOW})"),
    "babbage": ("text-babbage-001", f"BABBAGE (GPT-3, {WORKFLOW})"),
    "curie": ("text-curie-001", f"CURIE (GPT-3, {WORKFLOW})"),
    "davinci": ("text-davinci-003", f"DAVINCI (GPT-3, {WORKFLOW})"),
    "turbo": ("gpt-3.5-turbo", f"TURBO (GPT-3.5, {WORKFLOW})"),
}


TEST_SUITE_NAMES = [
    "CNN-DailyMail :: text length",
    "CNN-DailyMail :: news category",
    "CNN-DailyMail :: moderation score",
    "CNN-DailyMail :: text X ground truth length",
]

COMMON_METADATA = {
    "prompt": '"Professionally summarize this news article like a reporter with about '
    '{word_count_limit} to {word_count_limit+50} words :\n{full_text}"',
    "word_count_limit": "min(GT_wordcount - (GT_wordcount % 10), 380)",
    "fallback method": "delete half of the full text in the middle until it succeeds",
    "temperature": "0.4",
    "max_tokens": "400 or 300 (fallback)",
    "top_p": "1.0",
    "frequency_penalty": "0.0",
    "presence_penalty": "0.0",
}

MODEL_A = {
    "model family": "GPT-3",
    "model name": "text-ada-001",
    "description": "Very capable, faster and lower cost than Davinci.",
    "training cutoff": "Oct 2019",
}

MODEL_B = {
    "model family": "GPT-3",
    "model name": "text-babbage-001",
    "description": "Capable of straightforward tasks, very fast, and lower cost.",
    "training cutoff": "Oct 2019",
}

MODEL_C = {
    "model family": "GPT-3",
    "model name": "text-curie-001",
    "description": "Capable of very simple tasks, usually the fastest model in the GPT-3 series, and lowest cost.",
    "training cutoff": "Oct 2019",
}

MODEL_D = {
    "model family": "GPT-3",
    "model name": "text-davinci-003",
    "description": "Can do any language task with better quality, longer output,"
    + " and consistent instruction-following than the curie, babbage, or ada models.",
    "training cutoff": "Jun 2021",
}

MODEL_T = {
    "model family": "GPT-3.5",
    "model name": "gpt-3.5-turbo",
    "description": "Most capable GPT-3.5 model and optimized for chat at 1/10th the cost of text-davinci-003. "
    + "Will be updated with our latest model iteration.",
    "training cutoff": "Sep 2021",
}

MODEL_METADATA: Dict[str, Dict[str, str]] = {
    "text-ada-001": MODEL_A,
    "text-babbage-001": MODEL_B,
    "text-curie-001": MODEL_C,
    "text-davinci-003": MODEL_D,
    "gpt-3.5-turbo": MODEL_T,
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
            result = df.loc[df["article_id"] == sample.id]
            summary = result["prediction"].values[0]
            word_count = len(str(summary).split()) if not pd.isna(summary) else 0
            return Inference(
                summary=str(summary) if not pd.isna(summary) else "",
                is_failure=pd.isna(summary) or word_count <= 3,
                word_count=word_count,
                inference_time=float(result["prediction_time"].values[0]),
                tokens_input_text=int(result["tokens_text"].values[0]),
                tokens_ref_summary=int(result["tokens_summary"].values[0]),
                tokens_pred_summary=int(result["tokens_generated"].values[0]),
                tokens_prompt=int(result["tokens_used"].values[0]),
                cost=float(result["cost"].values[0]),
            )

        return infer

    model_name = mod[0]
    print(f"working on {model_name} and {test_suite.name} v{test_suite.version}")
    infer = get_stored_inferences(infs)
    model = Model(f"{mod[1]}", infer=infer, metadata={**MODEL_METADATA[model_name], **COMMON_METADATA})
    print(model)

    test(model, test_suite, evaluate_text_summarization, reset=True)


def seed_test(args: Namespace) -> None:
    mod = MODEL_MAP[args.model_name]
    print("loading inference CSV")
    s3_path = f"s3://kolena-public-datasets/CNN_DailyNews/results/{mod[0]}/results.csv"
    csv_to_use = s3_path if args.local_csv == "none" else args.local_csv
    columns_of_interest = [
        "article_id",
        "prediction",
        "prediction_time",
        "tokens_text",
        "tokens_summary",
        "tokens_generated",
        "tokens_used",
        "cost",
    ]
    df_results = pd.read_csv(csv_to_use, usecols=columns_of_interest)

    if args.test_suite == "none":
        print("loading test suite")
        test_suites = [TestSuite.load(name) for name in TEST_SUITE_NAMES]
        for test_suite in test_suites:
            print(test_suite.name)
            seed_test_run(mod, test_suite, df_results)
    else:
        test_suite = TestSuite.load(args.test_suite)
        print(test_suite.name)
        seed_test_run(mod, test_suite, df_results)


def main() -> None:
    ap = ArgumentParser()
    ap.add_argument("--model_name", type=str, help="One of 'ada', 'babbage', 'curie', 'davinci', or 'turbo'.")
    ap.add_argument("--test_suite", type=str, default="none", help="A specific test suite to run.", required=False)
    ap.add_argument("--local_csv", type=str, default="none", help="A specific csv to use.", required=False)
    kolena.initialize(os.environ["KOLENA_TOKEN"], verbose=True)

    seed_test(ap.parse_args())


if __name__ == "__main__":
    main()
