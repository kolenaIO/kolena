import re
import difflib
from numwords_to_nums.numwords_to_nums import NumWordsToNum

from typing import Union

from workflow import GroundTruth
from workflow import Inference

def generate_diff_word_level(reference: str, candidate: str, mode: str):
        '''
        Calculates the diff between the reference and candidate texts, and returns the following:
            output: text wrapped with false postive / false negative tags
            fp_count: # of false postive words
            fn_count: # of false negative words
            ins_count: # of insertions
            sub_count: # of substitutions
            del_count: # of deletions
            sub_list: List of GT words and their substitutions. Ex: marry → marie
            ins_list: List of inserted words
            del_list: List of deleted words
        '''
        matcher = difflib.SequenceMatcher(None, reference.split(), candidate.split())
        fp_count = 0
        fn_count = 0
        ins_count = 0
        sub_count = 0
        del_count = 0
        sub_list = []
        ins_list = []
        del_list = []
        
        output = []
        for opcode, a0, a1, b0, b1 in matcher.get_opcodes():
            if opcode == "equal":
                output.append(" ".join(matcher.a[a0:a1]))
                
            elif opcode == 'insert':
                fp_count += len(matcher.b[b0:b1])
                ins_count += len(matcher.b[b0:b1])
                ins_list.append(matcher.b[b0:b1])
                if mode == "fp":
                    output.append("<fp>" + " ".join(matcher.b[b0:b1]) + "</fp>")
                else:
                    output.append(" ".join(matcher.b[b0:b1]))

            elif opcode == 'delete':
                fn_count += len(matcher.a[a0:a1])
                del_count += len(matcher.a[a0:a1])
                del_list.append(matcher.a[a0:a1])
                if mode == "fn":
                    output.append("<fn>" + " ".join(matcher.a[a0:a1]) + "</fn>")
                else:
                    output.append(" ".join(matcher.a[a0:a1]))

            elif opcode == "replace":
                fn_count += len(matcher.a[a0:a1])
                fp_count += len(matcher.b[b0:b1])
                sub_count += len(matcher.a[a0:a1])
                sub_list.append(f"{' '.join(matcher.a[a0:a1])} → {' '.join(matcher.b[b0:b1])}")
                if mode == "fp":
                    output.append("<fp>" + " ".join(matcher.b[b0:b1]) + "</fp>")
                elif mode == "fn":
                    output.append("<fn>" + " ".join(matcher.a[a0:a1]) + "</fn>")
                else:
                    output.append(" ".join(matcher.b[b0:b1]))
        
        return " ".join(output), (fn_count, fp_count, ins_count, del_count, sub_count, sub_list, ins_list, del_list)


def preprocess_transcription(txt: Union[GroundTruth, Inference]):
    '''
    Preprocesses and standardizes text to prepare for metrics evaluations
    '''
    num = NumWordsToNum()
    txt = re.sub(r"[^\w\s]", "", txt.transcription.label.lower())
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