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
import re
from typing import List
from typing import Tuple

import numpy as np
import pandas as pd
from pyannote.core import Annotation
from pyannote.core import Segment
from scipy import optimize
from speaker_diarization.workflow import GroundTruth
from speaker_diarization.workflow import Inference
from speaker_diarization.workflow import TestCase

from kolena.workflow.annotation import TimeSegment

# allowed error (in seconds) when generating identification and missed speech errors.
ERROR_THRESHOLD = 0.2


def build_speaker_index(inf: List[Tuple[str, float, float]]) -> dict[str, int]:
    # borrowed from https://github.com/wq2012/SimpleDER/blob/master/simpleder/der.py
    speaker_set = sorted({element[0] for element in inf})
    index = {speaker: i for i, speaker in enumerate(speaker_set)}
    return index


def compute_intersection_length(A: List[Tuple[float, float]], B: List[Tuple[float, float]]) -> float:
    # borrowed from https://github.com/wq2012/SimpleDER/blob/master/simpleder/der.py
    max_start = max(A[1], B[1])
    min_end = min(A[2], B[2])
    return max(0.0, min_end - max_start)


def build_cost_matrix(ref: List[Tuple[str, float, float]], inf: List[Tuple[str, float, float]]) -> np.ndarray:
    # borrowed from https://github.com/wq2012/SimpleDER/blob/master/simpleder/der.py
    ref_index = build_speaker_index(ref)
    inf_index = build_speaker_index(inf)
    cost_matrix = np.zeros((len(ref_index), len(inf_index)))
    for ref_element in ref:
        for hyp_element in inf:
            i = ref_index[ref_element[0]]
            j = inf_index[hyp_element[0]]
            cost_matrix[i, j] += compute_intersection_length(
                ref_element,
                hyp_element,
            )
    return cost_matrix


def realign_labels(ref_df: pd.DataFrame, inf_df: pd.DataFrame):
    """
    Aligns speaker labels using linear sum optimiztion
    """
    ref = [(row.speaker, row.starttime, row.endtime) for i, row in ref_df.iterrows()]
    inf = [(row.speaker, row.starttime, row.endtime) for i, row in inf_df.iterrows()]
    cost_matrix = build_cost_matrix(ref, inf)
    row_index, col_index = optimize.linear_sum_assignment(-cost_matrix)

    ref_dict = build_speaker_index(ref)
    ref_dict_inv = {v: k for k, v in ref_dict.items()}
    inf_dict = build_speaker_index(inf)
    inf_dict_inv = {v: k for k, v in inf_dict.items()}

    mapping = {}
    for i in range(len(col_index)):
        mapping[inf_dict_inv[col_index[i]]] = ref_dict_inv[row_index[i]]

    inf_df["speaker"] = inf_df["speaker"].apply(lambda x: f"NA_{int(x)}" if x not in mapping.keys() else mapping[x])


def generate_annotation(segments: List[TimeSegment]) -> Annotation:
    """
    Generates pyannote Annotation objects for calculating metrics
    """
    annotation = Annotation()
    for segment in segments:
        annotation[Segment(segment.start, segment.end)] = segment.group

    return annotation


def preprocess_text(segments: List[TimeSegment]) -> str:
    """
    Preprocess text by removing punctuation and combining transcriptions.
    """
    text = " ".join([segment.label for segment in segments])
    text = re.sub(r"[^\w\s]", "", text.lower())
    return text


def create_non_overlapping_segments(
    transcription: List[TimeSegment],
    identity=None,
) -> List[Tuple[float, float]]:
    """
    Takes in a list of TimeSegments (overlapping or not) and returns a non-overlapping segment list.
    """
    res = []  # [(start, end), ...]
    transcription = sorted(transcription, key=lambda x: x.start)

    for t in transcription:
        if identity is not None and t.group != identity:
            continue

        start_time = t.start
        end_time = t.end

        start_idx = -1
        end_idx = -1
        for i, (s, e) in enumerate(res):
            if s <= start_time <= e:
                start_idx = i
            if s <= end_time <= e:
                end_idx = i

        if start_idx == end_idx and end_idx == -1:
            # add new segment to results
            res.append([start_time, end_time])
        elif start_idx != -1 and end_idx == -1:
            # extend interval by end_idx
            res[start_idx][1] = end_time
        elif start_idx == -1 and end_idx != -1:
            # extend interval by start_idx
            res[end_idx][0] = start_time
        elif start_idx < end_idx:
            # join two intervals
            res[start_idx][1] = res[end_idx][1]
            del res[end_idx]
        elif start_idx == end_idx:
            # interval already exists
            continue

    return res


def remove_intersection(inf: Tuple[float, float], gt: Tuple[float, float]) -> List[Tuple[float, float]]:
    """
    Takes in two intervals, GT and Inf, and removes the intersect of the two from inf.
    """
    inf_start, inf_end = inf
    gt_start, gt_end = gt

    if inf_end < gt_start or inf_start > gt_end:
        # No overlap
        return [inf]

    if inf_start < gt_start and inf_end > gt_end:
        # Return the two ends
        return [[inf_start, gt_start], [gt_end, inf_end]]
    elif inf_start >= gt_start and inf_end <= gt_end:
        # Entirely overlapped
        return None
    elif inf_start < gt_start:
        # Only return the left end
        return [[inf_start, gt_start]]
    else:
        # Only return the right end
        return [[gt_end, inf_end]]


def generate_error(gt: List[Tuple[float, float]], inf: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """
    Generates error intervals between a GT and Inf interval list.
    generate_error(gt, inf) returns the false positives, whereas generate_error(inf, gt) returns the false negatives.
    """
    res = inf.copy()

    i = 0
    while i != len(res):
        for j in range(len(gt)):
            removed = remove_intersection(res[i], gt[j])

            if removed is None:
                del res[i]
                i -= 1
                break
            else:
                del res[i]
                for r in reversed(removed):
                    res.insert(i, r)
        i += 1

    return res


def generate_identification_error(gt: GroundTruth, inf: Inference) -> List[TimeSegment]:
    """
    Highlights false positive speaker identifications.
    """
    unique_identities = set()
    for t in gt.transcription:
        unique_identities.add(t.group)

    res = []
    for id in unique_identities:
        gt_no = create_non_overlapping_segments(gt.transcription, id)
        inf_no = create_non_overlapping_segments(inf.transcription, id)
        res.extend(generate_error(gt_no, inf_no))
        res.extend(generate_error(inf_no, gt_no))

    res = [TimeSegment(start=r[0], end=r[1]) for r in res if r[1] - r[0] >= ERROR_THRESHOLD]

    return [TimeSegment(start=r[0], end=r[1]) for r in create_non_overlapping_segments(res)]


def invert_segments(segments: List[TimeSegment], end: float) -> List[Tuple[float, float]]:
    """
    Inverts time segments. Ex: [(start1, end1), (start2, end2)] -> [(end1, start2)]
    """
    res = []

    for i in range(len(segments) - 1):
        if segments[i + 1].start - segments[i].end > 0:
            res.append(segments[i + 1][0] - segments[i][1])

    if end - segments[-1][1] > 0:
        res.append(segments[-1][1], end)

    return res


def generate_missed_speech_error(gt: GroundTruth, inf: Inference) -> List[TimeSegment]:
    """
    Highlights all missed speech.
    """
    gt = create_non_overlapping_segments(gt.transcription)
    inf = create_non_overlapping_segments(inf.transcription)

    return [TimeSegment(start=r[0], end=r[1]) for r in generate_error(inf, gt)]


def calculate_tertiles(tc: TestCase, feature: str) -> dict:
    """
    Calculates the tertiles of a feature stored in metadata.
    """
    feature_list = [ts.metadata[feature] for ts, gt in tc.iter_test_samples()]
    percentiles = [np.percentile(feature_list, i) for i in np.linspace(0, 100, 4)]

    test_case_name_to_decision_logic_map = {
        "1st tertile": lambda x: x < percentiles[1],
        "2nd tertile": lambda x: percentiles[1] <= x < percentiles[2],
        "3rd tertile": lambda x: percentiles[2] <= x,
    }

    return test_case_name_to_decision_logic_map
