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
"""
Direct ViCLIP Video Embedding Extractor
This script extracts video embeddings using a locally downloaded ViCLIP model without
relying on the Hugging Face loading mechanism. It's designed to work around configuration
class mismatch errors that can occur with the standard loading approach.
The script creates a temporary package structure to handle relative imports in the model files,
loads the model weights directly from the safetensors file, and processes videos to extract
embeddings.
"""
import argparse
import json
import os
import pickle
import shutil
import sys
import tempfile
import warnings
from typing import Generator
from typing import List
from typing import Tuple

import cv2
import numpy as np
import torch
from tqdm import tqdm


class DirectViCLIPExtractor:
    """
    A class for extracting video embeddings using a locally downloaded ViCLIP model.
    This extractor creates a temporary package structure to handle relative imports
    in the model files and loads the model weights directly from the safetensors file.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        debug: bool = False,
    ) -> None:
        """
        Initialize the DirectViCLIPExtractor with a local ViCLIP model.
        Args:
            model_path (str): Path to the local ViCLIP model directory
            device (str): Device to run the model on ('cuda' or 'cpu')
            debug (bool): Enable debug mode with more verbose output
        """
        self.device = torch.device(device)
        self.model_path = model_path
        self.debug = debug

        # Image normalization constants
        self.v_mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
        self.v_std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)

        # Check vocabulary file and load model
        self._check_vocabulary_file()
        self._load_model()

    def _check_vocabulary_file(self) -> None:
        """Check if vocabulary file exists and copy it if needed."""
        vocab_file = os.path.join(self.model_path, "bpe_simple_vocab_16e6.txt.gz")
        if not os.path.exists(vocab_file):
            # Check if it exists in the current directory
            if os.path.exists("./bpe_simple_vocab_16e6.txt.gz"):
                print("Found vocabulary file in current directory, copying to model directory...")
                shutil.copy("./bpe_simple_vocab_16e6.txt.gz", vocab_file)
            else:
                raise FileNotFoundError(
                    f"Vocabulary file not found at {vocab_file} or in current directory. "
                    "Make sure to download it using download_viclip.py first.",
                )

    def _load_model(self) -> None:
        """Load the ViCLIP model from the local directory."""
        try:
            print(f"Loading model directly from {self.model_path}...")

            # Create a temporary package to handle relative imports
            self._setup_temp_package()

            # Load the configuration
            with open(os.path.join(self.model_path, "config.json")) as f:
                config_dict = json.load(f)

            if self.debug:
                print(f"Config: {config_dict}")

            # Import the model
            from viclip_temp_package.viclip import ViCLIP
            from transformers import PretrainedConfig

            # Create the config
            config = PretrainedConfig.from_dict(config_dict)

            # Set the tokenizer path
            config.tokenizer_path = os.path.join(self.model_path, "bpe_simple_vocab_16e6.txt.gz")

            # Create the model
            self.model = ViCLIP(config)

            # Load the weights
            self._load_model_weights()

            self.model.to(self.device)
            self.model.eval()
            print("Model loaded successfully")

        except Exception as e:
            print(f"Error loading model: {e}")
            if self.debug:
                import traceback

                traceback.print_exc()
            print("Make sure you have downloaded the model using download_viclip.py first.")
            sys.exit(1)

    def _load_model_weights(self) -> None:
        """Load model weights from safetensors file."""
        weights_path = os.path.join(self.model_path, "model.safetensors")
        if os.path.exists(weights_path):
            from safetensors.torch import load_file

            state_dict = load_file(weights_path)
            self.model.load_state_dict(state_dict)
            print("Model weights loaded from safetensors file")
        else:
            raise FileNotFoundError(f"Model weights not found at {weights_path}")

    def _setup_temp_package(self) -> None:
        """Create a temporary package to handle relative imports."""
        # Create a temporary directory for our package
        self.temp_dir = tempfile.mkdtemp()

        if self.debug:
            print(f"Created temporary directory: {self.temp_dir}")

        # Create the package directory
        package_dir = os.path.join(self.temp_dir, "viclip_temp_package")
        os.makedirs(package_dir, exist_ok=True)

        # Create an empty __init__.py file to make it a package
        with open(os.path.join(package_dir, "__init__.py"), "w") as f:
            f.write("# Temporary package for ViCLIP model\n")

        # Copy all Python files from the model directory to our package
        for file in os.listdir(self.model_path):
            if file.endswith(".py"):
                src = os.path.join(self.model_path, file)
                dst = os.path.join(package_dir, file)
                shutil.copy(src, dst)
                if self.debug:
                    print(f"Copied {src} to {dst}")

        # Add the temporary directory to the Python path
        sys.path.insert(0, self.temp_dir)

        if self.debug:
            print("Temporary package setup complete")

    def __del__(self) -> None:
        """Clean up temporary directory when the object is destroyed."""
        if hasattr(self, "temp_dir") and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            if self.debug:
                print(f"Removed temporary directory: {self.temp_dir}")

    def normalize(self, data: np.ndarray) -> np.ndarray:
        """
        Normalize image data.
        Args:
            data (np.ndarray): Input image data
        Returns:
            np.ndarray: Normalized image data
        """
        return (data / 255.0 - self.v_mean) / self.v_std

    def _frame_from_video(self, video: cv2.VideoCapture) -> Generator[np.ndarray, None, None]:
        """
        Extract frames from video.
        Args:
            video: OpenCV video capture object
        Yields:
            np.ndarray: Video frames
        """
        while video.isOpened():
            success, frame = video.read()
            if success:
                yield frame
            else:
                break

    def frames2tensor(
        self,
        vid_list: List[np.ndarray],
        fnum: int = 8,
        target_size: Tuple[int, int] = (224, 224),
    ) -> torch.Tensor:
        """
        Convert frames to tensor format required by ViCLIP.
        Args:
            vid_list (list): List of video frames
            fnum (int): Number of frames to sample
            target_size (tuple): Target size for resizing frames
        Returns:
            torch.Tensor: Tensor of frames in the format expected by ViCLIP
        """
        assert len(vid_list) >= fnum, f"Video has only {len(vid_list)} frames, but {fnum} are required"

        # Sample frames evenly
        step = len(vid_list) // fnum
        vid_list = vid_list[::step][:fnum]

        # Resize and convert to RGB
        vid_list = [cv2.resize(x[:, :, ::-1], target_size) for x in vid_list]

        # Normalize and convert to tensor
        vid_tube = [np.expand_dims(self.normalize(x), axis=(0, 1)) for x in vid_list]
        vid_tube = np.concatenate(vid_tube, axis=1)
        vid_tube = np.transpose(vid_tube, (0, 1, 4, 2, 3))
        vid_tube = torch.from_numpy(vid_tube).to(self.device, non_blocking=True).float()

        return vid_tube

    def get_vid_feat(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Get video features using ViCLIP.
        Args:
            frames (torch.Tensor): Tensor of video frames
        Returns:
            torch.Tensor: Video embedding features
        """
        with torch.no_grad():
            features = self.model.get_vid_features(frames)
        return features

    def extract_video_embedding(self, video_path: str, fnum: int = 8) -> np.ndarray:
        """
        Extract embedding for a single video file.
        Args:
            video_path (str): Path to the video file
            fnum (int): Number of frames to use for embedding
        Returns:
            numpy.ndarray: Video embedding
        """
        try:
            # Open video and extract frames
            video = cv2.VideoCapture(video_path)
            frames = [frame for frame in self._frame_from_video(video)]
            video.release()

            if len(frames) == 0:
                raise ValueError("No frames extracted from video")

            # Convert frames to tensor and get embedding
            frames_tensor = self.frames2tensor(frames, fnum=fnum)
            vid_feat = self.get_vid_feat(frames_tensor)
            vid_feat = vid_feat.cpu().numpy()

            return vid_feat

        except Exception as e:
            raise Exception(f"Error processing video {video_path}: {str(e)}")

    def process_video_folder(self, folder_path: str, output_pickle: str, fnum: int = 8) -> None:
        """
        Process all videos in a folder and save their embeddings.
        Args:
            folder_path (str): Path to folder containing videos
            output_pickle (str): Path to save the embeddings pickle file
            fnum (int): Number of frames to use per video
        """
        embeddings = {}
        video_files = [f for f in os.listdir(folder_path) if f.lower().endswith((".mp4", ".avi", ".mov"))]

        if not video_files:
            print("No video files found in the specified directory.")
            return

        for video_file in tqdm(video_files, desc="Processing videos"):
            video_path = os.path.join(folder_path, video_file)
            try:
                vid_feat = self.extract_video_embedding(video_path, fnum=fnum)
                embeddings[video_file] = vid_feat
            except Exception as e:
                print(f"Error processing {video_file}: {e}")

        # Save embeddings
        with open(output_pickle, "wb") as f:
            pickle.dump(embeddings, f)
        print(f"Embeddings successfully saved to {output_pickle}")


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract video embeddings using local ViCLIP model (direct loading)",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the local ViCLIP model directory",
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        required=True,
        help="Directory containing videos to process",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Output pickle file path for embeddings",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=8,
        help="Number of frames to sample from each video (default: 8)",
    )
    parser.add_argument(
        "--show_warnings",
        action="store_true",
        help="Show all warnings (including deprecation warnings from dependencies)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with more verbose output",
    )

    return parser.parse_args()


def main() -> None:
    """Main function to run the script."""
    args = parse_arguments()

    # Re-enable warnings if requested
    if args.show_warnings:
        warnings.resetwarnings()
        print("Warning display enabled")

    # Enable debug mode if requested
    debug_mode = args.debug
    if debug_mode:
        print("Debug mode enabled")
        import logging

        logging.basicConfig(level=logging.DEBUG)
        os.environ["TRANSFORMERS_VERBOSITY"] = "debug"

    # Validate paths
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model path not found: {args.model_path}")

    if not os.path.exists(args.video_dir):
        raise FileNotFoundError(f"Video directory not found: {args.video_dir}")

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)

    # Initialize extractor and process videos
    print(f"Initializing DirectViCLIPExtractor with model from {args.model_path}")
    extractor = DirectViCLIPExtractor(args.model_path, debug=debug_mode)

    print(f"Processing videos from {args.video_dir}")
    extractor.process_video_folder(args.video_dir, args.output_file, args.num_frames)

    print("\nExtraction complete! You can now upload the embeddings using upload_embeddings_to_kolena.py")


if __name__ == "__main__":
    main()
