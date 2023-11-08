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
import re
from typing import List
from typing import Set
from typing import Tuple

import numpy as np
import pandas as pd
from pyannote.core import Annotation
from pyannote.core import Segment
from scipy import optimize
from workflow import GroundTruth
from workflow import Inference

from kolena.workflow.annotation import LabeledTimeSegment
from kolena.workflow.annotation import TimeSegment

# allowed error (in seconds) when generating identification and missed speech errors.
ERROR_THRESHOLD = 0.2


def build_speaker_index(inf: List[Tuple[str, float, float]]) -> Set[str]:
    # borrowed from https://github.com/wq2012/SimpleDER/blob/master/simpleder/der.py
    speaker_set = sorted({element[0] for element in inf})
    index = {speaker: i for i, speaker in enumerate(speaker_set)}
    return index


def compute_intersection_length(A: List[Tuple[float, float]], B: List[Tuple[float, float]]) -> float:
    # borrowed from https://github.com/wq2012/SimpleDER/blob/master/simpleder/der.py
    max_start = max(A[1], B[1])
    min_end = min(A[2], B[2])
    return max(0.0, min_end - max_start)


def build_cost_matrix(ref, inf) -> dict:
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
    transcription: List[LabeledTimeSegment],
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
        return [inf]

    if inf_start < gt_start and inf_end > gt_end:
        return [[inf_start, gt_start], [gt_end, inf_end]]
    elif inf_start >= gt_start and inf_end <= gt_end:
        return None
    elif inf_start < gt_start:
        return [[inf_start, gt_start]]
    else:
        return [[gt_end, inf_end]]


def generate_fp(gt: List[Tuple[float, float]], inf: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """
    Generates false positive intervals between a GT and Inf interval list.
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

            if removed is not None:
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

        for fp in generate_fp(gt_no, inf_no):
            res.append(fp)

    i = 0
    while i != len(res):
        if res[i][1] - res[i][0] <= ERROR_THRESHOLD:
            del res[i]
            i -= 1
        i += 1

    return [LabeledTimeSegment(start=r[0], end=r[1], label="") for r in res]


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

    return [LabeledTimeSegment(start=r[0], end=r[1], label="") for r in generate_fp(inf, gt)]
