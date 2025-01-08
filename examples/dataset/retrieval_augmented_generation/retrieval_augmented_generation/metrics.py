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
import pandas as pd
from retrieval_augmented_generation.constants import ID_FIELDS


def is_doc_retrieved(retrieved_contents: list, doc_name: str) -> bool:
    return any([doc_name in content.locator for content in retrieved_contents])


def is_page_retrieved(retrieved_contents: list, doc_name: str, relevant_pages: list) -> bool:
    retrieved_pages = [content.page_number for content in retrieved_contents if doc_name in content.locator]

    # NOTE: all relevant pages must be retrieved to be considered correct.
    return set(relevant_pages).issubset(retrieved_pages)


def compute_metrics(df_dataset: pd.DataFrame, df_results: pd.DataFrame) -> pd.DataFrame:
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
