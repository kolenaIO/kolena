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
import random
import string
from argparse import Namespace
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

# Constants for testing
DATASET = "JAAD"
EMBEDDING_KEY = "viclip-embeddings"
MODEL_DIR = "examples/dataset/video_embedding_extraction/video_embedding_extraction/viclip_model"
EMBEDDINGS_FILE = "examples/dataset/video_embedding_extraction/video_embedding_extraction/embeddings.pkl"
VIDEO_DIR = "examples/dataset/video_embedding_extraction/video_embedding_extraction/videos"


@pytest.fixture(scope="module")
def dataset_name() -> str:
    TEST_PREFIX = "".join(random.choices(string.ascii_uppercase + string.digits, k=12))
    return f"{TEST_PREFIX} - {DATASET}"


@patch("os.path.exists", return_value=True)
def test__upload_embeddings__smoke(mock_exists: MagicMock, dataset_name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Simple smoke test for upload_embeddings_to_kolena"""
    from video_embedding_extraction.upload_embeddings_to_kolena import main

    # Mock the upload_dataset_embeddings function to avoid actual API calls
    monkeypatch.setattr(
        "video_embedding_extraction.upload_embeddings_to_kolena.upload_dataset_embeddings",
        lambda **kwargs: None,
    )

    # Mock load_embeddings to return an empty dict
    monkeypatch.setattr("video_embedding_extraction.upload_embeddings_to_kolena.load_embeddings", lambda file_path: {})

    # Mock create_embeddings_dataframe to return an empty DataFrame
    monkeypatch.setattr(
        "video_embedding_extraction.upload_embeddings_to_kolena.create_embeddings_dataframe",
        lambda embeddings, video_dir, s3_prefix: MagicMock(),
    )

    # Mock the argparse.ArgumentParser.parse_args to return our args
    monkeypatch.setattr(
        "argparse.ArgumentParser.parse_args",
        lambda self: Namespace(
            embeddings_file=EMBEDDINGS_FILE,
            video_dir=VIDEO_DIR,
            dataset_name=dataset_name,
            embedding_key=EMBEDDING_KEY,
            s3_prefix="s3://kolena-public-datasets/JAAD/JAAD_clips/",
        ),
    )

    # Call the main function
    main()


def test__download_viclip__smoke(monkeypatch: pytest.MonkeyPatch) -> None:
    """Simple smoke test for download_viclip"""
    from video_embedding_extraction.download_viclip import main

    # Mock the download functions to avoid actual downloads
    monkeypatch.setattr("video_embedding_extraction.download_viclip.download_vocab_file", lambda *args: None)
    monkeypatch.setattr("video_embedding_extraction.download_viclip.download_model_locally", lambda *args: None)

    # Mock os.makedirs to avoid creating directories
    monkeypatch.setattr("os.makedirs", lambda *args, **kwargs: None)

    # Mock the argparse.ArgumentParser.parse_args to return our args
    monkeypatch.setattr(
        "argparse.ArgumentParser.parse_args",
        lambda self: Namespace(
            output_dir=MODEL_DIR,
        ),
    )

    # Call the main function
    main()


@patch("os.path.exists", return_value=True)
def test__upload_embeddings_with_mocks(
    mock_exists: MagicMock,
    dataset_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test upload_embeddings_to_kolena with mocked dependencies"""
    from video_embedding_extraction.upload_embeddings_to_kolena import main
    import pandas as pd
    import numpy as np

    # Set up mock embeddings
    mock_embeddings = {"video1.mp4": np.random.rand(512)}

    # Set up mock DataFrame
    mock_df = pd.DataFrame(
        {
            "locator": ["s3://kolena-public-datasets/JAAD/JAAD_clips/video1.mp4"],
            "embedding": [np.random.rand(512)],
        },
    )

    # Mock the functions
    monkeypatch.setattr(
        "video_embedding_extraction.upload_embeddings_to_kolena.load_embeddings",
        lambda file_path: mock_embeddings,
    )
    monkeypatch.setattr(
        "video_embedding_extraction.upload_embeddings_to_kolena.create_embeddings_dataframe",
        lambda embeddings, video_dir, s3_prefix: mock_df,
    )

    # Mock the upload_dataset_embeddings function
    mock_upload = MagicMock()
    monkeypatch.setattr("video_embedding_extraction.upload_embeddings_to_kolena.upload_dataset_embeddings", mock_upload)

    # Mock the argparse.ArgumentParser.parse_args
    monkeypatch.setattr(
        "argparse.ArgumentParser.parse_args",
        lambda self: Namespace(
            embeddings_file=EMBEDDINGS_FILE,
            video_dir=VIDEO_DIR,
            dataset_name=dataset_name,
            embedding_key=EMBEDDING_KEY,
            s3_prefix="s3://kolena-public-datasets/JAAD/JAAD_clips/",
        ),
    )

    # Call the main function
    main()

    # Verify the upload function was called with the right arguments
    mock_upload.assert_called_once_with(
        dataset_name=dataset_name,
        key=EMBEDDING_KEY,
        df_embedding=mock_df,
    )


def test__download_viclip_with_mocks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test download_viclip with mocked dependencies"""
    from video_embedding_extraction.download_viclip import main

    # Set up mocks
    mock_makedirs = MagicMock()
    mock_download_vocab = MagicMock()
    mock_download_model = MagicMock()

    monkeypatch.setattr("os.makedirs", mock_makedirs)
    monkeypatch.setattr("video_embedding_extraction.download_viclip.download_vocab_file", mock_download_vocab)
    monkeypatch.setattr("video_embedding_extraction.download_viclip.download_model_locally", mock_download_model)

    # Mock the argparse.ArgumentParser.parse_args
    monkeypatch.setattr(
        "argparse.ArgumentParser.parse_args",
        lambda self: Namespace(
            output_dir=MODEL_DIR,
        ),
    )

    # Call the main function
    main()

    # Verify the mocks were called correctly
    mock_makedirs.assert_called_once_with(MODEL_DIR, exist_ok=True)
    mock_download_vocab.assert_called_once_with(MODEL_DIR)
    mock_download_model.assert_called_once_with(MODEL_DIR)
