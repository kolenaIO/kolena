# Copyright 2021-2025 Kolena Inc.
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
import pandas as pd
from retrieval_augmented_generation.constants import ID_FIELDS


def is_doc_retrieved(retrieved_contents: list, doc_names: list) -> bool:
    # Handle case where doc_names is a list
    return any(any(doc_name in content.locator for content in retrieved_contents) for doc_name in doc_names)


def is_page_retrieved(retrieved_contents: list, doc_names: list, relevant_pages: list) -> bool:
    # Create pairs of (doc_name, page_number) from retrieved contents
    retrieved_pairs = [
        (content.locator.split("/")[-1].replace(".pdf", ""), content.page_number) for content in retrieved_contents
    ]

    # Check if any of the relevant page pairs match with retrieved pairs
    return any(pair in retrieved_pairs for pair in relevant_pages)


def extract_doc_names(labeling_task: dict) -> list:
    if labeling_task is None:
        return []
    return [
        content.locator.split("/")[-1].replace(".pdf", "")
        for content in labeling_task.get("retrieved_contents", []) or []
    ]


def extract_relevant_pages(labeling_task: dict) -> list:
    if labeling_task is None:
        return []
    contents = labeling_task.get("retrieved_contents", []) or []
    # Create pairs of (doc_name, page_number)
    return [(content.locator.split("/")[-1].replace(".pdf", ""), content.page_number) for content in contents]


def compute_metrics(df_dataset: pd.DataFrame, df_results: pd.DataFrame) -> pd.DataFrame:
    ground_truth_columns = ["doc_names", "relevant_pages", "financebench_id"]

    is_labeled = "labeling_task" in df_dataset.columns
    if is_labeled:
        # Extract document names and page numbers from labeling task
        df_dataset["doc_names"] = df_dataset["labeling_task"].apply(extract_doc_names)
        df_dataset["relevant_pages"] = df_dataset["labeling_task"].apply(extract_relevant_pages)
    else:
        df_dataset["doc_names"] = df_dataset["doc_name"].apply(lambda x: [x])
        # Create pairs of (doc_name, page_number) for non-labeled data
        df_dataset["relevant_pages"] = df_dataset.apply(
            lambda row: [(row["doc_name"], page) for page in row["relevant_pages"]],
            axis=1,
        )
    assert set(ground_truth_columns).issubset(
        df_dataset.columns,
    ), f"ground truth columns {ground_truth_columns} cannot be found in dataset dataframe."
    # Include doc_names in the merge
    columns_to_merge = ground_truth_columns + ["doc_names"]
    df = df_results.merge(df_dataset[columns_to_merge], on=ID_FIELDS, how="left")

    metrics = []
    for record in df.itertuples():
        metrics.append(
            dict(
                is_doc_retrieved=is_doc_retrieved(record.retrieved_contents, record.doc_names),
                is_page_retrieved=is_page_retrieved(record.retrieved_contents, record.doc_names, record.relevant_pages),
            ),
        )

    return pd.DataFrame(metrics)
