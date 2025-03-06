# Example Integration: Video Embedding Extraction

This example integration demonstrates how to extract
video embeddings using the [ViCLIP](https://github.com/OpenGVLab/ViCLIP) model
and upload them to Kolena for video retrieval and analysis tasks.

## Setup

1. Ensure that data for the [`crossing pedestrian detection`](../crossing_pedestrian_detection)
dataset has been seeded through calling
the [`upload_dataset.py`](../crossing_pedestrian_detection/crossing_pedestrian_detection/upload_dataset.py)
 script.
2. This project uses [uv](https://docs.astral.sh/uv/) for packaging and
Python dependency management. To get started,
install project dependencies from [`pyproject.toml`](./pyproject.toml) by running:

```shell
uv sync
```

3. Download test videos to a local path for faster embedding extraction:

```shell
mkdir -p video_embedding_extraction/videos
aws s3 cp --recursive s3://kolena-public-examples/JAAD/data/sample_videos/ video_embedding_extraction/videos
```

## Usage

First, ensure that the `KOLENA_TOKEN` environment variable is populated
in your environment. See our
[initialization documentation](https://docs.kolena.com/installing-kolena/#initialization)
 for details.

This project defines three scripts that perform the following operations:

1. [`download_viclip.py`](./download_viclip.py) downloads the ViCLIP model and required vocabulary
 files locally.

2. [`video_embedding_extractor.py`](./video_embedding_extractor.py) extracts embeddings
from video files using the ViCLIP model.

3. [`upload_embeddings_to_kolena.py`](./upload_embeddings_to_kolena.py) uploads the extracted embeddings
 to a Kolena dataset.

### Step 1: Download the ViCLIP Model

First, download the ViCLIP model and vocabulary files:

```shell
uv run video_embedding_extraction/download_viclip.py
```

Optional command line arguments:

- `--output_dir`: Path to save the viclip model to

### Step 2: Extract Video Embeddings

Extract embeddings from a folder of video files using one of the extractor scripts:

```shell
uv run video_embedding_extraction/video_embedding_extractor.py
```

Optional command line arguments:

- `--model_path`: Path to the downloaded ViCLIP model
- `--video_dir`: Directory containing video files to process
- `--output_file`: Path to save the embeddings pickle file
- `--num_frames`: Number of frames to sample from each video (default: 8)

### Step 3: Upload Embeddings to Kolena

Upload the extracted embeddings to a Kolena dataset:

```shell
uv run video_embedding_extraction/upload_embeddings_to_kolena.py
```

Optional command line arguments:

- `--embeddings_file`: Path to the pickle file containing video embeddings
- `--video_dir`: Directory containing the video files (for verification)
- `--dataset_name`: Name of the existing Kolena dataset to upload embeddings to
- `--embedding_key`: Unique identifier for these embeddings
