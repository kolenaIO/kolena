# Copyright 2021-2025 Kolena Inc.
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
from typing import Set

import torch
from transformers import AutoModelForTokenClassification
from transformers import AutoTokenizer

PII_MODEL_NAME = "iiiorg/piiranha-v1-detect-personal-information"

tokenizer = AutoTokenizer.from_pretrained(PII_MODEL_NAME)
model = AutoModelForTokenClassification.from_pretrained(PII_MODEL_NAME)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def detect_pii_in_dataframe() -> None:
    pass


def detect_pii_in_string(text: str, allowed_pii_types: Set[str] = set()) -> bool:
    # Tokenize input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Get the model predictions
    with torch.no_grad():
        outputs = model(**inputs)

    # Get the predicted labels
    predictions = torch.argmax(outputs.logits, dim=-1)

    # Convert token predictions to word predictions
    encoded_inputs = tokenizer.encode_plus(text, return_offsets_mapping=True, add_special_tokens=True)
    offset_mapping = encoded_inputs["offset_mapping"]

    is_pii = False
    pii_data_start = 0
    pii_data_end = 0
    pii_type = ""

    for i, (start, end) in enumerate(offset_mapping):
        if start == end:  # Special token
            continue

        pred_id = predictions[0][i].item()
        if pred_id != model.config.label2id["O"] and model.config.id2label[pred_id] not in allowed_pii_types:
            current_pii_type = model.config.id2label[pred_id]
            if not is_pii:
                is_pii = True
                pii_data_start = start
                pii_type = current_pii_type
            elif pii_type == current_pii_type:
                pii_data_end = end + 1
            elif pii_type != current_pii_type:
                # break out of the loop if the current pii data has finished scanning
                break
        elif is_pii:
            break

    if is_pii:
        print(f"[{pii_type}] data detected: {text[pii_data_start:pii_data_end]}")
        return True
    return False
