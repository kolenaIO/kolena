---
icon: kolena/flame-16
---

# :kolena-flame-20: Quickstart

Install Kolena to set up rigorous and repeatable model testing in minutes.

In this quickstart guide, we will use the [300 Faces In the Wild (300-W)](https://ibug.doc.ic.ac.uk/resources/300-W/)
example dataset to demonstrate the how to curate test data and test models in Kolena, using both the web app and
the `kolena` Python SDK.

??? note "Using the SDK"
    If you would like to use the SDK for this quickstart follow these instructions to set up the
    SDK and set up the code example locally.

    First ,install the `kolena` python SDK:

    === "`pip`"

        ```shell
        pip install kolena
        ```

    === "`poetry`"

        ```shell
        poetry add kolena
        ```

    Then, clone the example code:

    ```shell
    git clone https://github.com/kolenaIO/kolena.git
    ```

    The [kolenaIO/kolena](https://github.com/kolenaIO/kolena) repository contains a number of example integrations to clone
    and run directly. Set up the keypoint detection example for this quickstart:

    ```shell
    cd kolena/examples/dataset/keypoint_detection
    poetry update && poetry install
    ```

## Step 1: Upload dataset

Model evaluations on Kolena starts with datasets. Datasets are tables that contain the data you wish to use for
creating test cases.

=== "Web App"

    To import the 300-W dataset, select “Import datasets” then “Select from cloud storage”. Using the explorer, navigate to
    s3://kolena-pubic-examples/300-W/ and select 300-W.csv.

    ??? note "Generating Datasets"
        See the [`keypoint_detetction/upload_dataset.py`](https://github.com/kolenaIO/kolena/blob/trunk/examples/dataset/keypoint_detection/keypoint_detection/upload_dataset.py)
        script in the code example for details on how the dataset is were generated.

    Note: The keypoints in `300-W.csv` have been defined using `kolena.workflow.annotation.Keypoints` from
    the `kolena` SDK. See the `keypoint_detetction/upload_dataset.py` script for example usage.

    You will now see a preview of how the information is going to be consumed by Kolena.

    Give your dataset a name and select `locator` as the ID column. The ID column is used to uniquely identify a test
    sample and is used when uploading model results to associate inferences with test samples.

    Click “Confirm” to import the dataset. Once the dataset has been added, you can add description and tags to them to organize the workspace.

    <figure markdown>
        ![Example Dataset Upload](../assets/images/quickstart-upload-dataset-workflow.gif)
        <figcaption>Example Dataset Upload</figcaption>
    </figure>


=== "SDK"

    The example code contains a script [`keypoint_detetction/upload_dataset.py`](https://github.com/kolenaIO/kolena/blob/trunk/examples/dataset/keypoint_detection/keypoint_detection/upload_dataset.py)
    which will process the CSV file `s3://kolena-public-examples/300-W/raw/300-W.csv`
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
    see in [:kolena-dataset-20: Datasets](https://app.kolena.io/redirect/datasets).


## Step 2: Upload Model Results

Model results are supplied as tables containing the ID column chosen when uploading the dataset. For
this example, we will upload the results for the open-source [RetinaFace](https://github.com/serengil/retinaface) (`RetinaFace` option)
keypoint detection model and a random keypoint model.

=== "Web App"

    To upload new model results, from the Details tab of the dataset, click on “Upload Model Results” in the upper right.
    Then, select “Upload from cloud storage”. Using the explorer, navigate to s3://kolena-public-datasets/300-W/results/
    and select **model.csv**. This CSV file contains model results for each
    of the datapoints we uploaded to the dataset.

    ??? note "Generating Model Results"
        See the [`keypoint_detetction/upload_results.py`](https://github.com/kolenaIO/kolena/blob/trunk/examples/dataset/keypoint_detection/keypoint_detection/upload_results.py)
        script in the code example for details on how results were generated.

    You will now see a preview of how Kolena will ingest the model results. Give your model a name, and click “Confirm” to
    upload the model results.

    **gif of model import workflow**

    Repeat the above steps with the file **model/other-results.csv**.

=== "SDK"

    The example code contains a script [`keypoint_detetction/upload_results.py`](https://github.com/kolenaIO/kolena/blob/trunk/examples/dataset/keypoint_detection/keypoint_detection/upload_results.py)
    which will compute metrics on a CSV of model inferences and upload them to Kolena.

    ```shell
    poetry run python3 keypoint_detection/upload_results.py RetinaFace
    poetry run python3 keypoint_detection/upload_results.py random
    ```

    Results for two models named "RetinaFace" and "random" should now appear <somewhere>

## Step 3: Explore Data and Results

Once you have uploaded your dataset and model results, you can visualize the data using Kolena’s plotting tools.

You can quickly see the distribution of any datapoint or model inference field in the “Distributions” tab.

Additionally, you can create custom plots in the "Debugger" tab. For example, click “Add Model” in the top left and
select the RetinaFace model. In the plotting widget at the bottom, select **something** as the x-axis
and **something else** as the y-axis. You will then see a plot showing **something**

**gif of plot generation**

## Step 4: Define Quality Standards

Quality Standards define the criteria by which models are evaluated on each dataset. A Quality Standard consists of
Test Cases, which organize your data into key scenarios, and Evaluation Criteria, which define key performance metrics.

### Define Test Cases

To configure test cases, navigate to the “Debugger” tab and click on “Configure Test Cases”. From here, you can select a
field to use stratify your dataset. For this example, divide the dataset by <>. Click “Save to Quality
Standards” to save your test case configuration.

You will now see that your dataset has been organized into test cases based on the category field. Any Evaluation
Criteria you define will be calculated on each test case.

**gif of test case creation**

### Define Evaluation Criteria

To configure Evaluation Criteria, navigate to the “Debugger” tab and click on “Configure Evaluation Criteria”.
In the drawer, click on “Add Evaluation Criteria”, and select **suggested evaluation criteria**

**gif of evaluation criteria definition**

## Step 5: Compare Models

Once you have configured a Quality Standard, the Evaluation Metrics you define will be calculated across all Test Cases.
To compare the results across models, navigate to the "Quality Standards" tab and add the "RetinaFace" and "random" models to the
table using the + button in the top right. You will now see all evaluation criteria in the quality standard computed
on every test case in the quality standard.

**gif of model comparison**

## Conclusion

In this quickstart, we used an example integration from [kolenaIO/kolena](https://github.com/kolenaIO/kolena) to
integrate data from the [300 Faces In the Wild (300-W)](https://ibug.doc.ic.ac.uk/resources/300-W/) dataset,
created test cases, and tested the open source open-source [RetinaFace](https://github.com/serengil/retinaface)
keypoint detection model.

This example shows us how to quickly and easily integrate test data and evaluate model performance in Kolena,
and can be arbitrarily extended with additional metrics, plots, visualizations, and data.
