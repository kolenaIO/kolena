# :octicons-flame-24: Quickstart

Install Kolena to set up rigorous and repeatable model testing in minutes.

## Install and Initialize `kolena`

Install the `kolena` Python package to programmatically interact with Kolena:

=== "`pip`"

    ```shell
    pip install kolena
    ```

=== "`poetry`"

    ```shell
    poetry add kolena
    ```

With the client installed, visit [app.kolena.io/~/developer](https://app.kolena.io/redirect/developer) to generate an
API token. Copy and paste the code snippet to set the `KOLENA_TOKEN` environment variable:

```shell
export KOLENA_TOKEN="********"
```

Now initialize a session with `kolena.initialize`:

```python
import os
import kolena

kolena.initialize(os.environ["KOLENA_TOKEN"], verbose=True)
```

## Clone an Example

The [kolenaIO/kolena](https://github.com/kolenaIO/kolena) repository contains a number of example integrations to clone
and run directly:

<div class="grid cards" markdown>
- [Example: Keypoint Detection](https://github.com/kolenaIO/kolena/tree/trunk/examples/keypoint_detection)

    ![image](assets/images/300-W.jpg)

    ---

    Facial Keypoint Detection using the [300 Faces in the Wild (300-W)](https://ibug.doc.ic.ac.uk/resources/300-W/)
    dataset

- [Example: Age Estimation](https://github.com/kolenaIO/kolena/tree/trunk/examples/age_estimation)

    ![image](assets/images/LFW.jpg)

    ---

    Age Estimation using the [Labeled Faces in the Wild (LFW)](http://vis-www.cs.umass.edu/lfw/) dataset

- [Example: Text Summarization](https://github.com/kolenaIO/kolena/tree/trunk/examples/text_summarization)

    ![image](assets/images/CNN-DailyMail.jpg)

    ---

    Text Summarization using [OpenAI GPT-family models](https://platform.openai.com/docs/guides/gpt) and the
    [CNN-DailyMail](https://paperswithcode.com/dataset/cnn-daily-mail-1) dataset
</div>
