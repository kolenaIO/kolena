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
from scipy import optimize
from workflow import GroundTruth
from workflow import Inference

from kolena.workflow.annotation import LabeledTimeSegment


def build_speaker_index(inf):
    # borrowed from https://github.com/wq2012/SimpleDER/blob/master/simpleder/der.py
    speaker_set = sorted({element[0] for element in inf})
    index = {speaker: i for i, speaker in enumerate(speaker_set)}
    return index


def compute_intersection_length(A, B):
    # borrowed from https://github.com/wq2012/SimpleDER/blob/master/simpleder/der.py
    max_start = max(A[1], B[1])
    min_end = min(A[2], B[2])
    return max(0.0, min_end - max_start)


def build_cost_matrix(ref, inf):
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


def realign_labels(ref_df, inf_df):
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


def calc_overlap(gt_start, gt_end, inf_start, inf_end):
    start = max(gt_start, inf_start)
    end = min(gt_end, inf_end)
    return {
        "start": start,
        "end": end,
        "dist": end - start,
    }


def generate_tp(gt: GroundTruth, inf: Inference, identification=False):
    gt_idx = 0
    inf_idx = 0

    res = []
    while gt_idx < len(gt.transcription) and inf_idx < len(inf.transcription):
        gt_t = gt.transcription[gt_idx]
        inf_t = inf.transcription[inf_idx]

        if gt_t.start > inf_t.end:
            inf_idx += 1
            continue
        if inf_t.start > gt_t.end:
            gt_idx += 1
            continue

        if inf_t.end > gt_t.end:
            if not identification or inf_t.group == gt_t.group:
                res.append(LabeledTimeSegment(start=max(inf_t.start, gt_t.start), end=gt_t.end, label=""))
            gt_idx += 1

        elif inf_t.end == gt_t.end:
            if not identification or inf_t.group == gt_t.group:
                res.append(LabeledTimeSegment(start=max(inf_t.start, gt_t.start), end=gt_t.end, label=""))
            gt_idx += 1
            inf_idx += 1

        elif inf_t.end < gt_t.end:
            if not identification or inf_t.group == gt_t.group:
                res.append(LabeledTimeSegment(start=max(inf_t.start, gt_t.start), end=inf_t.end, label=""))
            inf_idx += 1

    return res


# going to remove this in the final PR
def generate_fp(gt: GroundTruth, inf: Inference, identification=False):
    gt_idx = 0
    inf_idx = 0

    res = []
    while gt_idx < len(gt.transcription) and inf_idx < len(inf.transcription):
        gt_t = gt.transcription[gt_idx]
        inf_t = inf.transcription[inf_idx]

        if gt_t.start > inf_t.end:
            inf_idx += 1
            continue
        if inf_t.start > gt_t.end:
            gt_idx += 1
            continue

        if inf_t.end > gt_t.end:
            if inf_t.start >= gt_t.start:
                if gt_idx != len(gt.transcription) - 1:
                    res.append(
                        LabeledTimeSegment(gt_t.end, end=min(gt.transcription[gt_idx + 1].start, inf_t.end), label=""),
                    )  #
                else:
                    res.append(LabeledTimeSegment(gt_t.end, end=inf_t.end, label=""))

                if identification and inf_t.group != gt_t.group:
                    res.append(LabeledTimeSegment(inf_t.start, end=gt_t.end, label=""))
            else:
                if gt_idx != 0:
                    res.append(
                        LabeledTimeSegment(
                            start=max(inf_t.start, gt.transcription[gt_idx - 1].end),
                            end=gt_t.start,
                            label="",
                        ),
                    )
                else:
                    res.append(LabeledTimeSegment(start=inf_t.start, end=gt_t.start, label=""))

                if gt_idx != len(gt.transcription) - 1:
                    res.append(
                        LabeledTimeSegment(gt_t.end, end=min(gt.transcription[gt_idx + 1].start, inf_t.end), label=""),
                    )
                else:
                    res.append(LabeledTimeSegment(gt_t.end, end=inf_t.end, label=""))

                if identification and inf_t.group != gt_t.group:
                    res.append(LabeledTimeSegment(gt_t.start, end=gt_t.end, label=""))
            gt_idx += 1

        elif inf_t.end == gt_t.end:
            if inf_t.start < gt_t.start:
                if gt_idx != 0:
                    res.append(
                        LabeledTimeSegment(
                            start=max(inf_t.start, gt.transcription[gt_idx - 1].end),
                            end=gt_t.start,
                            label="",
                        ),
                    )
                else:
                    res.append(LabeledTimeSegment(start=inf_t.start, end=gt_t.start, label=""))

                if identification and inf_t.group != gt_t.group:
                    res.append(LabeledTimeSegment(inf_t.start, end=gt_t.end, label=""))

            gt_idx += 1
            inf_idx += 1

        elif inf_t.end < gt_t.end:
            if inf_t.start < gt_t.start:
                if gt_idx != 0:
                    res.append(
                        LabeledTimeSegment(
                            start=max(inf_t.start, gt.transcription[gt_idx - 1].end),
                            end=gt_t.start,
                            label="",
                        ),
                    )
                else:
                    res.append(LabeledTimeSegment(start=inf_t.start, end=gt_t.start, label=""))

                if identification and inf_t.group != gt_t.group:
                    res.append(LabeledTimeSegment(gt_t.start, end=inf_t.end, label=""))

            inf_idx += 1

    for i, sample in enumerate(res):
        if sample.start >= sample.end:
            del res[i]
        elif sample.end - sample.start <= 0.2:
            del res[i]

    return res


def inv(gt: GroundTruth, inf: Inference):
    endpoint = gt.transcription[-1].end
    res = []

    ts = inf.transcription
    for i in range(len(ts) - 1):
        if ts[i + 1].start - ts[i].end > 0:
            if i == 0 or i == len(ts) - 2 or not ((ts[i - 1].end >= ts[i].end) or (ts[i + 2].start <= ts[i + 1].start)):
                res.append(LabeledTimeSegment(start=ts[i].end, end=ts[i + 1].start, label=""))

    if endpoint - ts[-1].end > 0:
        res.append(LabeledTimeSegment(start=ts[-1].end, end=endpoint, label=""))

    return Inference(transcription=res)


# work in progress
def create_non_overlapping_segments(transcription: List[LabeledTimeSegment]):
    res = []  # [(start, end), ...]
    transcription = sorted(transcription, key=lambda x: x.start)

    for t in transcription:
        start_time = t.start
        end_time = t.end

        start_idx = -1
        end_idx = -1
        for i, (s, e) in enumerate(res):
            if s <= start_time <= e:
                start_idx = i
            if s <= end_time <= e:
                end_idx = i

        assert end_idx == start_idx or end_idx - start_idx == 1

        if start_idx == end_idx and end_idx == -1:
            # create own segment
            pass
        elif start_idx != -1 and end_idx == -1:
            # extend interval by end_idx
            pass
        elif start_idx == -1 and end_idx != -1:
            # extend interval by start_idx
            pass
        elif start_idx < end_idx:
            # join two intervals
            pass
        elif start_idx == end_idx:
            # interval already exists
            continue


# work in progress
def generate_identification_error(gt: GroundTruth, inf: Inference):
    pass
