# Copyright 2021-2023 Kolena Inc.
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
import os

import pandas as pd
from activation_map.utils import bitmap_locator  # noqa: F401
from activation_map.utils import create_bitmap
from activation_map.utils import load_activation_map
from activation_map.utils import upload_bitmap  # noqa: F401

BUCKET = "kolena-public-datasets"
FOLDER = "advanced-usage/uploading-activation-map/"
META_DIR = "meta"
BITMAP_DIR = "bitmap-masks"


def main() -> None:
    metadata_filepath = os.path.join("s3://", BUCKET, FOLDER, META_DIR, "activation_maps.csv")
    df_metadata = pd.read_csv(metadata_filepath)

    for record in df_metadata.itertuples(index=False):
        filename = os.path.splitext(os.path.basename(record.activation_map_locator))[0]  # noqa: F841
        activation_map = load_activation_map(record.activation_map_locator)
        image_buffer = create_bitmap(activation_map)  # noqa: F841

        # commenting out this uploading section due to read-only access granted on the S3 BUCKET.
        # suggest replacing this section with `bitmap_locator` pointing to your own cloud storage.
        # locator = bitmap_locator(BUCKET, os.path.join(FOLDER, BITMAP_DIR), filename)
        # upload_bitmap(image_buffer, locator)


if __name__ == "__main__":
    main()
