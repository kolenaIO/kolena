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
import difflib
import re
from typing import Any
from typing import Dict
from typing import Union

from automatic_speech_recognition.workflow import GroundTruth
from automatic_speech_recognition.workflow import Inference
from numwords_to_nums.numwords_to_nums import NumWordsToNum


def generate_diff_word_level(reference: str, candidate: str) -> Dict[str, Any]:
    """
    Calculates the diff between the reference and candidate texts, and returns the following:
        fp_str: text wrapped with false postive tags
        fn_str: text wrapped with false negative tags
        fp_count: # of false postive words
        fn_count: # of false negative words
        ins_count: # of insertions
        sub_count: # of substitutions
        del_count: # of deletions
        sub_list: List of GT words and their substitutions. Ex: marry → marie
        ins_list: List of inserted words
        del_list: List of deleted words
    """
    matcher = difflib.SequenceMatcher(None, reference.split(), candidate.split())
    fp_count = 0
    fn_count = 0
    ins_count = 0
    sub_count = 0
    del_count = 0
    sub_list = []
    ins_list = []
    del_list = []

    fp_str = []
    fn_str = []
    for opcode, ref_start, ref_end, can_start, can_end in matcher.get_opcodes():
        if opcode == "equal":
            fp_str.append(" ".join(matcher.a[ref_start:ref_end]))   # type: ignore
            fn_str.append(" ".join(matcher.a[ref_start:ref_end]))   # type: ignore

        elif opcode == "insert":
            fp_count += len(matcher.b[can_start:can_end])   # type: ignore
            ins_count += len(matcher.b[can_start:can_end])   # type: ignore
            ins_list.append(matcher.b[can_start:can_end])   # type: ignore
            fp_str.append("<fp>" + " ".join(matcher.b[can_start:can_end]) + "</fp>")   # type: ignore
            fn_str.append(" ".join(matcher.b[can_start:can_end]))   # type: ignore

        elif opcode == "delete":
            fn_count += len(matcher.a[ref_start:ref_end])   # type: ignore
            del_count += len(matcher.a[ref_start:ref_end])   # type: ignore
            del_list.append(matcher.a[ref_start:ref_end])   # type: ignore
            fn_str.append("<fn>" + " ".join(matcher.a[ref_start:ref_end]) + "</fn>")   # type: ignore
            fp_str.append(" ".join(matcher.a[ref_start:ref_end]))   # type: ignore

        elif opcode == "replace":
            fn_count += len(matcher.a[ref_start:ref_end])   # type: ignore
            fp_count += len(matcher.b[can_start:can_end])   # type: ignore
            sub_count += len(matcher.a[ref_start:ref_end])   # type: ignore
            sub_list.append(
                f"{' '.join(matcher.a[ref_start:ref_end])}"   # type: ignore
                f" → {' '.join(matcher.b[can_start:can_end])}",  # type: ignore
            )
            fp_str.append("<fp>" + " ".join(matcher.b[can_start:can_end]) + "</fp>")  # type: ignore
            fn_str.append("<fn>" + " ".join(matcher.a[ref_start:ref_end]) + "</fn>")  # type: ignore

    return {
        "fp_str": " ".join(fp_str),
        "fn_str": " ".join(fn_str),
        "fn_count": fn_count,
        "fp_count": fp_count,
        "ins_count": ins_count,
        "del_count": del_count,
        "sub_count": sub_count,
        "sub_list": sub_list,
        "ins_list": ins_list,
        "del_list": del_list,
    }


def preprocess_transcription(txt: str) -> str:
    """
    Preprocesses and standardizes text to prepare for metrics evaluations.
    Removes punctuation, changes text to lower case, and converts all NumWords to Numbers.
    """
    num = NumWordsToNum()
    txt = re.sub(r"[^\w\s]", "", txt)
    return "oh".join(
        [
            num.numerical_words_to_numbers(
                "th".join(
                    [num.numerical_words_to_numbers(x, convert_operator=True) for x in re.split(r"(?<=[a-zA-Z])th", y)],
                ),
                convert_operator=True,
            )
            for y in txt.split("oh")
        ],
    )
