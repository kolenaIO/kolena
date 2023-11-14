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
from typing import List

import numpy as np
from recommender_system.workflow import TestSampleMetrics

from kolena.workflow import Histogram

# CONSTANTS
GENRES = [
    "Adventure",
    "Animation",
    "Children's",
    "Comedy",
    "Fantasy",
    "Romance",
    "Drama",
    "Action",
    "Crime",
    "Thriller",
    "Horror",
    "Mystery",
    "Sci-Fi",
    "IMAX",
    "Documentary",
    "War",
    "Musical",
    "Western",
    "Film-Noir",
    "(no genres listed)",
]

ID_OCCUPATION_MAP = {
    0: "other",
    1: "academic/educator",
    2: "artist",
    3: "clerical/admin",
    4: "college/grad student",
    5: "customer service",
    6: "doctor/health care",
    7: "executive/managerial",
    8: "farmer",
    9: "homemaker",
    10: "K-12 student",
    11: "lawyer",
    12: "programmer",
    13: "retired",
    14: "sales/marketing",
    15: "scientist",
    16: "self-employed",
    17: "technician/engineer",
    18: "tradesman/craftsman",
    19: "unemployed",
    20: "writer",
}

ID_AGE_MAP = {1: "Under 18", 18: "18-24", 25: "25-34", 35: "35-44", 45: "45-49", 50: "50-55", 56: "56+"}


# Stratification Categories
PS = "Professional Services"
MB = "Management and Business"
SSI = "Support and Service Industry"
ES = "Education and Students"
CA = "Creative and Artistic"
M = "Miscellaneous"
L25 = "Under 25"
B25_50 = "25-50"
U50 = "50+"

# Stratifications Mapping
OCCUPATION_STRATIFICATION = {
    "academic/educator": PS,
    "artist": CA,
    "clerical/admin": M,
    "college/grad student": ES,
    "customer service": SSI,
    "doctor/health care": PS,
    "executive/managerial": MB,
    "farmer": SSI,
    "homemaker": SSI,
    "K-12 student": ES,
    "lawyer": PS,
    "programmer": PS,
    "retired": M,
    "sales/marketing": MB,
    "scientist": PS,
    "self-employed": MB,
    "technician/engineer": M,
    "tradesman/craftsman": SSI,
    "unemployed": M,
    "writer": CA,
    "other": M,
}

AGE_STRATIFICATION = {
    "Under 18": L25,
    "18-24": L25,
    "25-34": B25_50,
    "35-44": B25_50,
    "45-49": B25_50,
    "50-55": U50,
    "56+": U50,
}


def process_metadata(record, f):
    value = getattr(record, f)
    if f == "genres":
        return value.split("|")
    elif f == "age":
        return ID_AGE_MAP[value]
    elif f == "occupation":
        return ID_OCCUPATION_MAP[value]

    return value


# Plot Functions
def create_histogram(
    metrics: List[TestSampleMetrics],
) -> Histogram:
    deltas = [tsm.Δ_rating for tsm in metrics]
    min_data, max_data = -5.0, 5.0

    number_of_bins = 50
    bin_size = (max_data - min_data) / number_of_bins
    bin_edges = [min_data + i * bin_size for i in range(number_of_bins + 1)]

    freq, bin_edges = np.histogram(deltas, bins=bin_edges, density=True)

    return Histogram(
        title="Delta Rating Distribution",
        x_label="Δ_rating (inf - gt)",
        y_label="Frequency (%)",
        buckets=list(bin_edges),
        frequency=list(freq),
    )
