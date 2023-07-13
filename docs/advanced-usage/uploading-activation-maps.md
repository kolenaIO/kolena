---
icon: kolena/heatmap-16
---

# :kolena-heatmap-20: Uploading Activation Maps

As AI models advance and become more complex, humans can no longer understand the reasoning behind the decisions or
predictions made by the AI. There are many advantages on understanding how our model had predicted a specific output,
and the process called **Explainable AI** (XAI) can help AI scientists and developers comprehend and trust the model
results. There are many explanation methods for different types of AI model architecture, and most of the popular
techniques used in computer vision workflows output a map that highlights regions in an image that are relevant to the
model output, and this map is called **activation map**.

<figure markdown>
  ![activation map](../assets/images/activation-map.png)
  <figcaption markdown>Visualization of an activation map overlaid on an image</figcaption>
</figure>

## Can I Visualize Activation Maps on Kolena?

Yes! Activation maps can be visualized as an overlay on the corresponding image in
[:kolena-studio-16: Studio](https://app.kolena.io/redirect/studio) using an
[`Annotation`][kolena.workflow.annotation.Annotation] type called [`BitmapMask`][kolena.workflow.annotation.BitmapMask]
which can help us understand the model’s decision — what the model “sees” when it makes its prediction.

In this tutorial, we’ll learn how to upload activation maps on Kolena.

## How to Upload Activation Maps on Kolena?

Uploading activation maps to Kolena can be done in three simple steps:

- [**Step 1**](#step-1-creating-png-bitmaps): creating PNG bitmaps from 2D array activation maps
- [**Step 2**](#step-2-uploading-png-bitmaps-to-cloud): uploading in-memory PNG bitmaps to a cloud storage
- [**Step 3**](#step-3-updating-inference-and-running-tests): updating inferences and running tests

Let's take a look at each step with example code snippets.

### Step 1: Creating PNG Bitmaps

The activation map is a 2D data array ranging from 0 to 1 with `(h, w)` shape. This array is converted to a PNG bitmap
using the following two utility methods:

- [`colorize_activation_map`][kolena.workflow.visualization.colorize_activation_map]: applies color and opacity to the input activation map
- [`create_png`][kolena.workflow.visualization.create_png]: creates an in-memory PNG image represented as binary data

```python
import io
import numpy as np
from kolena.workflow.visualization import colorize_activation_map
from kolena.workflow.visualization import create_png

def create_bitmap(activation_map: np.ndarray) -> io.BytesIO:
    bitmap = colorize_activation_map(activation_map)
    image_buffer = create_png(bitmap, mode="RGBA")
    return image_buffer
```

!!! info "**Activation Map Scaling**"
    The activation map often has the equal dimensions (i.e., width and height) as the input image or sometimes has the
    scaled-down dimensions with the fixed ratio. Kolena automatically scales the overlay annotations to the images so
    there is no need to up-scale the map to match the image dimensions.

### Step 2: Uploading PNG Bitmaps to Cloud

In order to visualize the bitmaps on Kolena, these bitmaps must be uploaded to a cloud storage first, and their locators
are used to create [`BitmapMask`][kolena.workflow.annotation.BitmapMask]s. In this tutorial, we will learn how to upload
the in-memory bitmaps to a S3 bucket. For any other cloud storage type, please refer to your cloud storage's Python
API docs.

```python
import io
import boto3
from urllib.parse import urlparse

BUCKET = <YOUR_S3_BUCKET>

s3 = boto3.client("s3")

def bitmap_locator(filename: str) -> str:
    return f"{BUCKET}/tutorial/activation_maps/{filename}.png"

def upload_bitmap(image_buffer: io.BytesIO, filename: str) -> str:
    locator = bitmap_locator(filename)
    parsed_url = urlparse(locator)
    s3_bucket = parsed_url.netloc
    s3_key = parsed_url.path.lstrip("/")
    s3.upload_fileobj(image_buffer, s3_bucket, s3_key)
    return locator
```

With all the building blocks we learned from [Step 1](#step-1-creating-png-bitmaps) and
[Step 2](#step-2-uploading-png-bitmaps-to-cloud), we can now create a
[`BitmapMask`][kolena.workflow.annotation.BitmapMask] with a given activation map.

```python
def create_and_upload_bitmap(
    filename: str,
    activation_map: np.ndarray,
) -> BitmapMask:
    image_buffer = create_bitmap(activation_map)
    locator = upload_bitmap(image_buffer, filename)
    return BitmapMask(locator)
```

### Step 3: Updating `Inference` and Running tests

!!! info inline end
    If you are not familiar with the workflow concept, please read the
    [:kolena-cube-20: Building a Workflow](https://docs.kolena.io/building-a-workflow) guide.

For the purposes of this tutorial, let's assume we already have a workflow built, and we are going to upload
the activation maps as one of the fields in [`Inference`](https://docs.kolena.io/building-a-workflow/#inference-type).
All we need to do is to update the `Inference` definition to include a new field for the activation map:

```python
from kolena.workflow import Inference as Inf
from kolena.workflow.annotation import BitmapMask

@dataclass(frozen=True)
class Inference(Inf):
    ...
    activation_map: BitmapMask
```

!!! info inline end
    If you are not familiar with how to run tests, please read the
    [Step 4: Running Tests](https://docs.kolena.io/building-a-workflow/#step-4-running-tests)
    from [:kolena-cube-20: Building a Workflow](https://docs.kolena.io/building-a-workflow) guide.

Before you run tests, make sure to update your `infer` function to return an `Inference` with the corresponding
`BitmapMask` as its `activation_map` field. You are now ready to run tests! Once the tests complete, we can now visit
[:kolena-studio-16: Studio](https://app.kolena.io/redirect/studio) to visualize activation maps overlaid on your
[`Image`][kolena.workflow.Image] data.

## Conclusion
In this tutorial, we learned how to upload activation maps to Kolena in order to visualize activation maps
overlaid on your [`Image`][kolena.workflow.Image] data along with your ground truths and inferences.
