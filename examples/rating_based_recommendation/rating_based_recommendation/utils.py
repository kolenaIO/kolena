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

ID_AGE_MAP = {1: "Under 18", 18: "18-24", 25: "25-34", 35: "35-44", 45: "45-49", 50: "50-55", 56: "56+"}


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
        x_label="Δ_rating",
        y_label="Frequency (%)",
        buckets=list(bin_edges),
        frequency=list(freq),
    )
