import numpy as np
from typing import List

from rating_based_recommendation.workflow import TestSampleMetrics
from kolena.workflow import Histogram

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

PS = "Professional Services"
MB = "Management and Business"
SSI = "Support and Service Industry"
ES = "Education and Students"
CA = "Creative and Artistic"
M = "Miscellaneous"

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

ID_AGE_MAP = {1: "Under 18", 18: "18-24", 25: "25-34", 35: "35-44", 45: "45-49", 50: "50-55", 56: "56+"}

L25 = "Under 25"
B25_50 = "25-50"
U50 = "50+"

AGE_STRATIFICATION = {
    "Under 18": L25,
    "18-24": L25,
    "25-34": B25_50,
    "35-44": B25_50,
    "45-49": B25_50,
    "50-55": U50,
    "56+": U50,
}


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
        x_label="Δ_rating (predicted rating - real rating)",
        y_label="Frequency (%)",
        buckets=list(bin_edges),
        frequency=list(freq),
    )
