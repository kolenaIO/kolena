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
import warnings

import pandas as pd


def upload_dataset_embeddings(dataset_name: str, key: str, df_embedding: pd.DataFrame) -> None:
    """
    Upload a list of search embeddings for a dataset.

    .. deprecated:: 1.55.0
        Use :func:`kolena.dataset.search.upload_dataset_embeddings` instead.

    :param dataset_name: String value indicating the name of the dataset for which the embeddings will be uploaded.
    :param key: String value uniquely corresponding to the embedding vectors. For example, this can be the name of the
        embedding model along with the column with which the embedding was extracted, such as `resnet50-image_locator`.
    :param df_embedding: Dataframe containing id fields for identifying datapoints in the dataset and the associated
        embeddings as `numpy.typing.ArrayLike` of numeric values.
    :raises NotFoundError: The given dataset does not exist.
    :raises InputValidationError: The provided input is not valid.
    """
    warnings.warn(
        "\n kolena._experimental.search.upload_dataset_embeddings is deprecated. \n"
        " Use kolena.dataset.search.upload_dataset_embeddings instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    # Import here to avoid circular dependency
    from kolena.dataset.search import upload_dataset_embeddings as new_upload_dataset_embeddings

    new_upload_dataset_embeddings(dataset_name, key, df_embedding)
