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
from named_entity_recognition.metrics import evaluate
from named_entity_recognition.metrics import find_overlap

from kolena.annotation import LabeledTextSegment

segment1 = LabeledTextSegment(text_field="field1", start=10, end=20, label="label1")
segment2 = LabeledTextSegment(text_field="field1", start=20, end=30, label="label2")
segment3 = LabeledTextSegment(text_field="field1", start=30, end=40, label="label1")
# matches with segment1
segment_tp = LabeledTextSegment(
    text_field="field1",
    start=10,
    end=20,
    label="label1",
    score=0.9,  # type: ignore[call-arg]
)
# loc matches with segment2, bad cls
segment_fp_cls = LabeledTextSegment(
    text_field="field1",
    start=20,
    end=30,
    label="label1",
    score=0.5,  # type: ignore[call-arg]
)
# partial match with segment3, bad localization
segment_fp_loc = LabeledTextSegment(
    text_field="field1",
    start=31,
    end=40,
    label="label1",
    score=0.5,  # type: ignore[call-arg]
)
# complete mismatch
segment_fp_full = LabeledTextSegment(
    text_field="field1",
    start=100,
    end=200,
    label="label1",
    score=0.5,  # type: ignore[call-arg]
)


def test__find_overlap() -> None:
    overlap = find_overlap(
        [segment1],
        [segment1],
    )
    assert overlap == {segment1}

    overlap = find_overlap(
        [segment1],
        [segment2],
    )
    assert overlap == set()

    overlap = find_overlap(
        [segment1],
        [segment3],
    )
    assert overlap == set()

    overlap = find_overlap(
        [segment1, segment2],
        [segment1, segment3],
    )
    assert overlap == {segment1}

    overlap = find_overlap(
        [segment1, segment2],
        [segment2, segment1],
    )
    assert overlap == {segment1, segment2}


def test__evaluate() -> None:
    gts = [
        segment1,
        segment2,
        segment3,
    ]
    predictions = [
        segment_tp,
        segment_fp_cls,
        segment_fp_loc,
        segment_fp_full,
    ]
    tp, fp, fn, metrics = evaluate(gts, predictions, ["label1", "label2"])
    assert tp == [segment1]
    assert fp == [segment_fp_cls, segment_fp_loc, segment_fp_full]
    assert fn == [segment2, segment3]
    assert metrics == dict(
        TP=len(tp),
        FP=len(fp),
        FN=len(fn),
        GT=len(gts),
        INF=len(predictions),
        LOC_ERROR=1,
        CLS_ERROR=1,
        OVR_GEN=1,
    )
