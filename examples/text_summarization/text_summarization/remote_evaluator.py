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
import os

from text_summarization.evaluator_fast import evaluate_text_summarization_fast
from text_summarization.workflow import Model
from text_summarization.workflow import TestSuite

import kolena
from kolena.workflow import test

# envvars set by Kolena remote evaluator entry code
model_name = os.environ["KOLENA_MODEL_NAME"]
test_suite_name = os.environ["KOLENA_TEST_SUITE_NAME"]
test_suite_version = os.environ["KOLENA_TEST_SUITE_VERSION"]
env_token = "KOLENA_TOKEN"

print(f"initializing with environment variables ${env_token}")
kolena.initialize(os.environ[env_token], verbose=True)

model = Model(model_name)
print(f"using model: {model}")
test_suite = TestSuite.load(test_suite_name, version=test_suite_version)
print(f"using test_suite: {test_suite}")

print("running evaluation...")
test(model, test_suite, evaluate_text_summarization_fast)
print("finished running evaluation")
