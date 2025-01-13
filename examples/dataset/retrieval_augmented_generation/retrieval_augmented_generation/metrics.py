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
    # Get all retrieved pages for any matching document
    retrieved_pages = [
        content.page_number for content in retrieved_contents for doc_name in doc_names if doc_name in content.locator
    ]

    # NOTE: all relevant pages must be retrieved to be considered correct.
    return set(relevant_pages).issubset(retrieved_pages)


def compute_metrics(df_dataset: pd.DataFrame, df_results: pd.DataFrame) -> pd.DataFrame:
    is_labeled = "labeling_task" in df_dataset.columns
    if is_labeled:
        # Extract all unique document names from all retrieved contents' locators
        df_dataset["doc_name"] = df_dataset["labeling_task"].apply(
            lambda x: list(
                {
                    content.locator.split("/")[-1].replace(".pdf", "")
                    for content in x.get("retrieved_contents", []) or []
                },
            )
            if x is not None
            else [],
        )
        # Extract page numbers from retrieved contents
        df_dataset["relevant_pages"] = df_dataset["labeling_task"].apply(
            lambda x: [content.page_number for content in x.get("retrieved_contents", []) or []]
            if x is not None
            else [],
        )
        ground_truth_columns = ["doc_name", "relevant_pages", "financebench_id"]
    else:
        ground_truth_columns = ["doc_name", "relevant_pages", "financebench_id"]
        assert set(ground_truth_columns).issubset(
            df_dataset.columns,
        ), f"ground truth columns {ground_truth_columns} cannot be found in dataset dataframe."

    df = df_results.merge(df_dataset[ground_truth_columns], on=ID_FIELDS, how="left")

    metrics = []
    for record in df.itertuples():
        metrics.append(
            dict(
                is_doc_retrieved=is_doc_retrieved(record.retrieved_contents, record.doc_name),
                is_page_retrieved=is_page_retrieved(record.retrieved_contents, record.doc_name, record.relevant_pages),
            ),
        )

    return pd.DataFrame(metrics)
