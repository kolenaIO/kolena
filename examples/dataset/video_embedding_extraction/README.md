# Example Integration: Video Embedding Extraction

This example integration demonstrates how to extract video embeddings using the [ViCLIP](https://github.com/OpenGVLab/ViCLIP) model and upload them to Kolena for video retrieval and analysis tasks.

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for packaging and Python dependency management. To get started,
install project dependencies from [`pyproject.toml`](./pyproject.toml) by running:

```shell
uv sync
```

## Usage

First, ensure that the `KOLENA_TOKEN` environment variable is populated in your environment. See our
[initialization documentation](https://docs.kolena.com/installing-kolena/#initialization) for details.

This project defines three scripts that perform the following operations:

1. [`download_viclip.py`](./download_viclip.py) downloads the ViCLIP model and required vocabulary files locally.

2. [`video_embedding_extractor.py`](./video_embedding_extractor.py) extracts embeddings from video files using the ViCLIP model.

3. [`upload_embeddings_to_kolena.py`](./upload_embeddings_to_kolena.py) uploads the extracted embeddings to a Kolena dataset.

### Step 1: Download the ViCLIP Model

First, download the ViCLIP model and vocabulary files:

```shell
uv run download_viclip.py --output_dir ./viclip_model
```

### Step 2: Extract Video Embeddings

Extract embeddings from a folder of video files using one of the extractor scripts:

#### Option 1: Using the standard extractor (offline mode only)

```shell
uv run video_embedding_extractor.py --model_path ./viclip_model --video_dir ./videos --output_file ./embeddings.pkl --num_frames 8
```

#### Option 2: Using the direct extractor (recommended for configuration class mismatch errors)

```shell
uv run direct_video_embedding_extractor.py --model_path ./viclip_model --video_dir ./videos --output_file ./embeddings.pkl --num_frames 8
```

Command line arguments:

- `--model_path`: Path to the downloaded ViCLIP model
- `--video_dir`: Directory containing video files to process
- `--output_file`: Path to save the embeddings pickle file
- `--num_frames`: Number of frames to sample from each video (default: 8)
- `--show_warnings`: Show all warnings (including deprecation warnings from dependencies)
- `--debug`: Enable debug mode with more verbose output (only available in direct_video_embedding_extractor.py)

### Step 3: Upload Embeddings to Kolena

Upload the extracted embeddings to a Kolena dataset:

```shell
uv run upload_embeddings_to_kolena.py --embeddings_file ./embeddings.pkl --video_dir ./videos --dataset_name "Joint Attention in Autonomous Driving (JAAD)" --embedding_key "viclip-embeddings"
```

Command line arguments:

- `--embeddings_file`: Path to the pickle file containing video embeddings
- `--video_dir`: Directory containing the video files (for verification)
- `--dataset_name`: Name of the existing Kolena dataset to upload embeddings to
- `--embedding_key`: Unique identifier for these embeddings
