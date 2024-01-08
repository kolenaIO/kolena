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
from commons import load_data
from commons import DATASET
from commons import BUCKET

import kolena
from kolena.dataset import upload_dataset


def main() -> None:
    kolena.initialize(verbose=True)
    df_metadata_csv = pd.read_csv(f"s3://{BUCKET}/{DATASET}/meta/metadata_complete.csv", storage_options={"anon": True})
    df_metadata = load_data(df_metadata_csv[:100], is_pred=False)
    upload_dataset(DATASET, df_metadata)


if __name__ == "__main__":
    main()