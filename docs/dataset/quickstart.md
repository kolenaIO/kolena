---
icon: kolena/flame-16
---

# :kolena-flame-20: Quickstart

Install Kolena to set up rigorous and repeatable model testing in minutes.

In this quickstart guide, we'll use we will use the [300 Faces In the Wild (300-W)](https://ibug.doc.ic.ac.uk/resources/300-W/)
example dataset to demonstrate the how to curate test data and test models in Kolena. The examples
use data located in a public S3 bucket `s3://kolena-public-examples`

## Installing `kolena` Python SDK

=== "`pip`"

    ```shell
    pip install kolena
    ```

=== "`poetry`"

    ```shell
    poetry add kolena
    ```

## Clone the examples

The [kolenaIO/kolena](https://github.com/kolenaIO/kolena) repository contains a number of example integrations to clone
and run directly:

<div class="grid cards" markdown>

- [:kolena-keypoint-detection-20: Example: Keypoint Detection ↗](https://github.com/kolenaIO/kolena/tree/trunk/examples/dataset/keypoint_detection)

    ![Example image and five-point facial keypoints array from the 300 Faces in the Wild dataset.](../assets/images/300-W.jpg)

    ---

    End-to-end face recognition 1:1 using the [Labeled Faces in the Wild (LFW)](http://vis-www.cs.umass.edu/lfw/) dataset.
- [:kolena-chat-20: Example: Question Answering ↗](https://github.com/kolenaIO/kolena/tree/trunk/examples/dataset/question_answering)

    ![Example questions from CoQA dataset.](../assets/images/CoQA.jpg)

    ---

    Question Answering using the
    [Conversational Question Answering (CoQA)](https://stanfordnlp.github.io/coqa/) dataset
</div>

Now, let's set up the keypoint detection example:

```shell
cd kolena/examples/object_detection_2d
poetry update && poetry install
```

## Dataset Upload

Model evaluations on Kolena starts with datasets. Datasets are tables that contain the data you wish to use for
creating test cases.

=== "SDK"

    The example code contains a script `keypoint_detection/upload_dataset.py` which will process the CSV
    file `s3://kolena-public-examples/300-`
    and register it as a dataset in Kolena using the `register_dataset` function.

    First, let's first configure our environment by populating the `KOLENA_TOKEN`
    environment variable. Visit the [:kolena-developer-16: Developer](https://app.kolena.io/redirect/developer) page to
    generate an API token and copy and paste the code snippet into your environment:

    ```shell
    export KOLENA_TOKEN="********"
    ```

    We can now register a new dataset using the provided script:

    ```shell
    poetry run python3 keypoint_detection/upload_dataset.py
    ```

    After this script has completed, a new dataset named "300-W" will be created, which you can
    see in [:kolena-dataset-20: Datasets](https://app.kolena.io/redirect/datasets)

=== "Web App"

    To import the 300-W dataset, select “Import datasets” then “Select from cloud storage”. Using the explorer, navigate to
    s3://kolena-pubic-examples/300-W/ and select 300-W.csv.

    Note: The keypoints in `300-W.csv` have been defined using `kolena.workflow.annotation.Keypoints` from
    the `kolena` SDK. See the `keypoint_detetction/upload_dataset.py` script for example usage.

    You will now see a preview of how the information is going to be consumed by Kolena.

    Give your dataset a name and select `locator` as the ID column. The ID column is used to uniquely identify a test
    sample and is used when uploading model results to associate inferences with test samples.

    Click “Confirm” to import the dataset. Once the dataset has been added, you can add description and tags to them to organize the workspace.

    *gif of dataset import workflow*


## Model Upload

Model inferences are supplied as tables containing the ID column chosen when uploading the dataset. For
this example, we will upload the results for the open-source [RetinaFace](https://github.com/serengil/retinaface) (`RetinaFace` option)
keypoint detection model and a random keypoint model.

=== "SDK"

    The example code contains a script `keypoint_detection/upload_results.py` which will compute metrics on
    a CSV of model inferences and upload them to Kolena.

    ```shell
    poetry run python3 keypoint_detection/upload_results.py RetinaFace
    poetry run python3 keypoint_detection/upload_results.py random
    ```

    Results for two models named "RetinaFace" and "random" should now appear <somewhere>

=== "Web App"

    To upload new model inferences, click on “Update” > “Upload Model Results” from the Details tab of the dataset.
    Then, select “Upload from cloud storage”. Using the explorer, navigate to s3://kolena-public-datasets/300-W/results/
    and select <model/results.csv>. This CSV file contains model inferences and metrics for each
    of the datapoints we uploaded to the dataset.

    Note: See the `keypoint_detetction/upload_results.py` script for details on how inferences and
    metrics were generated.

    You will now see a preview of how Kolena will ingest the model results. Give your model a name, and click “Confirm” to
    upload the model results.

    **gif of model import workflow**

    Repeat the above steps with the file **model/other-results.csv**.

## Visualization

Once you have uploaded your dataset and model results, you can visualize the data using Kolena’s plotting tools.

You can quickly see the distribution of any datapoint or model inference field in the “Distributions” tab.

Additionally, you can create custom plots in the “Results” tab. For example, click “Add Model” in the top left and
select the RetinaFace model. In the plotting widget at the bottom, select **something** as the x-axis
and **something else** as the y-axis. You will then see a plot showing **something**

**gif of plot generation**

## Quality Standards

Quality Standards define the criteria by which models are evaluated on each dataset. A Quality Standard consists of
Test Cases, which organize your data into key scenarios, and Evaluation Criteria, which define key performance metrics.

### Test Cases

To configure test cases, navigate to the “Results” tab and click on “Configure Test Cases”. From here, you can select a
field to use stratify your dataset. For this example, divide the dataset by <>. Click “Save to Quality
Standards” to save your test case configuration.

You will now see that your dataset has been organized into test cases based on the category field. Any Evaluation
Criteria you define will be calculated on each test case.

**gif of test case creation**

### Evaluation Criteria

To configure Evaluation Criteria, navigate to the “Results” tab and click on “Configure Evaluation Criteria”.
In the drawer, click on “Add Evaluation Criteria”, and select <><><>

**gif of evaluation criteria definition**

## Compare Models

Once you have configured a Quality Standard, the Evaluation Metrics you define will be calculated across all Test Cases.
To compare the results across models, navigate to the “Model Cards” tab and add the <><><> models to the
table using the + button in the top right. You will now see all evaluation criteria in the quality standard computed
on every test case in the quality standard.

## Conclusion

In this quickstart, we used an example integration from [kolenaIO/kolena](https://github.com/kolenaIO/kolena) to integrate data from the
[300 Faces In the Wild (300-W)](https://ibug.doc.ic.ac.uk/resources/300-W/) dataset, created test cases,
and tested the open source open-source [RetinaFace](https://github.com/serengil/retinaface) keypoint detection model.

This example shows us how to quickly and easily integrate test data and evaluate model performance in Kolena,
and can be arbitrarily extended with additional metrics, plots, visualizations, and data.
