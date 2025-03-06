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
# Dataset and upload constants
S3_LOCATOR_PREFIX = "s3://kolena-public-examples/JAAD/data/videos/"
DATASET_NAME = "JAAD [crossing-pedestrian-detection]"
EMBEDDING_KEY = "viclip-embeddings"

# ViCLIP model constants
VICLIP_VOCAB_URL = "https://openaipublic.azureedge.net/clip/bpe_simple_vocab_16e6.txt.gz"
VICLIP_MODEL_NAME = "OpenGVLab/ViCLIP-L-14-hf"
DEFAULT_MODEL_OUTPUT_DIR = "./viclip_model"
BPE_VOCAB_FILE = "bpe_simple_vocab_16e6.txt.gz"

# Video processing constants
DEFAULT_FRAME_COUNT = 8
DEFAULT_EMBEDDINGS_FILE = "./video_embedding_extraction/embeddings.pkl"
DEFAULT_VIDEO_DIR = "./video_embedding_extraction/videos"
