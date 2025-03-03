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
import argparse
import os
import shutil
import sys

import requests
from transformers import AutoConfig
from transformers import AutoModel


def download_vocab_file(output_dir: str) -> None:
    """
    Download the BPE vocabulary file required by ViCLIP.

    Args:
        output_dir (str): Directory to save the vocabulary file
    """
    vocab_url = "https://openaipublic.azureedge.net/clip/bpe_simple_vocab_16e6.txt.gz"
    vocab_path = os.path.join(output_dir, "bpe_simple_vocab_16e6.txt.gz")

    print(f"Downloading vocabulary file from {vocab_url}...")
    try:
        response = requests.get(vocab_url, stream=True)
        response.raise_for_status()  # Raise an exception for HTTP errors

        with open(vocab_path, "wb") as f:
            shutil.copyfileobj(response.raw, f)

        print(f"Vocabulary file saved to {vocab_path}")

        if not os.path.exists("./bpe_simple_vocab_16e6.txt.gz"):
            print("Creating symlink to vocabulary file in current directory...")
            os.symlink(vocab_path, "./bpe_simple_vocab_16e6.txt.gz")
            print("Symlink created")
    except Exception as e:
        print(f"Error downloading vocabulary file: {e}")
        sys.exit(1)


def download_model_locally(output_dir: str) -> None:
    """
    Download the ViCLIP model locally.

    Args:
        output_dir (str): Directory to save the model
    """
    print("Downloading ViCLIP model...")
    try:
        config = AutoConfig.from_pretrained("OpenGVLab/ViCLIP-L-14-hf", trust_remote_code=True)
        print("Successfully downloaded model configuration")

        # Then download the model
        model = AutoModel.from_pretrained(
            "OpenGVLab/ViCLIP-L-14-hf",
            trust_remote_code=True,
            config=config,
        )
        model.save_pretrained(output_dir)
        print(f"Model saved to {output_dir}")
    except Exception as e:
        print(f"Error downloading model: {e}")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download ViCLIP model and vocabulary file for video embedding extraction",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./viclip_model",
        help="Directory to save the model and vocabulary file (default: ./viclip_model)",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    download_vocab_file(args.output_dir)

    download_model_locally(args.output_dir)

    print("\nDownload complete! You can now use the model with video_embedding_extractor.py")
    print(f"Model and vocabulary files are saved in: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
