---
icon: kolena/flame-16
---

# :kolena-flame-20: Quickstart

Install Kolena to set up rigorous and repeatable model testing in minutes.

In this quickstart guide, we'll use the
[`age_estimation`](https://github.com/kolenaIO/kolena/tree/trunk/examples/age_estimation) example integration to
demonstrate the how to curate test data and test models in Kolena.

## Install `kolena`

Install the `kolena` Python package to programmatically interact with Kolena:

=== "`pip`"

    ```shell
    pip install kolena
    ```

=== "`poetry`"

    ```shell
    poetry add kolena
    ```

## Clone the Examples

The [kolenaIO/kolena](https://github.com/kolenaIO/kolena) repository contains a number of example integrations to clone
and run directly:

<div class="grid cards" markdown>
- [:kolena-age-estimation-20: Example: Age Estimation](https://github.com/kolenaIO/kolena/tree/trunk/examples/age_estimation)

    ![Example images from the Labeled Faces in the Wild dataset.](assets/images/LFW.jpg)

    ---

    Age Estimation using the [Labeled Faces in the Wild (LFW)](http://vis-www.cs.umass.edu/lfw/) dataset

- [:kolena-keypoint-detection-20: Example: Keypoint Detection](https://github.com/kolenaIO/kolena/tree/trunk/examples/keypoint_detection)

    ![Example image and five-point facial keypoints array from the 300 Faces in the Wild dataset.](assets/images/300-W.jpg)

    ---

    Facial Keypoint Detection using the [300 Faces in the Wild (300-W)](https://ibug.doc.ic.ac.uk/resources/300-W/)
    dataset

- [:kolena-text-summarization-20: Example: Text Summarization](https://github.com/kolenaIO/kolena/tree/trunk/examples/text_summarization)

    ![Example articles from the CNN-DailyMail dataset.](assets/images/CNN-DailyMail.jpg)

    ---

    Text Summarization using [OpenAI GPT-family models](https://platform.openai.com/docs/guides/gpt) and the
    [CNN-DailyMail](https://paperswithcode.com/dataset/cnn-daily-mail-1) dataset
</div>

To get started, clone the `kolena` repository:

```shell
git clone https://github.com/kolenaIO/kolena.git
```

With the repository cloned, let's set up the
[`age_estimation`](https://github.com/kolenaIO/kolena/tree/trunk/examples/age_estimation) example:

```shell
cd kolena/examples/age_estimation
poetry update && poetry install
```

Now we're up and running and can start [creating test suites](#create-test-suites) and
[testing models](#test-a-model).

## Create Test Suites

Each of the example integrations comes with scripts for two flows:

1. `seed_test_suite.py`: Create test cases and test suite(s) from a source dataset
2. `seed_test_run.py`: Test model(s) on the created test suites

Before running [`seed_test_suite.py`](https://github.com/kolenaIO/kolena/blob/trunk/examples/age_estimation/age_estimation/seed_test_suite.py),
let's first configure our environment by populating the `KOLENA_TOKEN`
environment variable. Visit the [:kolena-developer-16: Developer](https://app.kolena.io/redirect/developer) page to
generate an API token and copy and paste the code snippet into your environment:

```shell
export KOLENA_TOKEN="********"
```

We can now create test suites using the provided seeding script:

```shell
poetry run python3 age_estimation/seed_test_suite.py
```

After this script has completed, we can visit the [:kolena-test-suite-16: Test Suites](https://app.kolena.io/redirect/testing)
page to view our newly created test suites.

In this `age_estimation` example, we've created test suites stratifying the LFW dataset (which is stored as a CSV in
S3) into test cases by age, estimated race, and estimated gender.

## Test a Model

After we've created test suites, the final step is to test models on these test suites. The `age_estimation` example
provides the `ssrnet` model for this step:

```shell
poetry run python3 age_estimation/seed_test_run.py \
  "ssrnet" \
  "age :: labeled-faces-in-the-wild [age estimation]" \
  "race :: labeled-faces-in-the-wild [age estimation]" \
  "gender :: labeled-faces-in-the-wild [age estimation]"
```

!!! note "Note: Testing additional models"
    In this example, model results have already been extracted and are stored in CSV files in S3. To run a new model,
    plug it into the `infer` method in [`seed_test_run.py`](https://github.com/kolenaIO/kolena/blob/trunk/examples/age_estimation/age_estimation/seed_test_run.py).

Once this script has completed, click the results link in your console or visit
[:kolena-results-16: Results](https://app.kolena.io/redirect/results) to view the test results for this newly tested model.

## Conclusion

In this quickstart, we used an example integration from [kolenaIO/kolena](https://github.com/kolenaIO/kolena) to create
test suites from the [Labeled Faces in the Wild (LFW)](http://vis-www.cs.umass.edu/lfw/) dataset and test the
open-source `ssrnet` model on these test suites.

This example shows us how to define an ML problem as a workflow for testing in Kolena, and can be arbitrarily extended
with additional metrics, plots, visualizations, and data.
