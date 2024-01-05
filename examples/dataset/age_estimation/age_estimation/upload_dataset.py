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
from age_estimation.constants import DATA_FILEPATH
from age_estimation.constants import DATASET

import kolena
from kolena.dataset import register_dataset


def main() -> None:
    df = pd.read_csv(DATA_FILEPATH)

    kolena.initialize(verbose=True)
    register_dataset(DATASET, df)


if __name__ == "__main__":
    main()
