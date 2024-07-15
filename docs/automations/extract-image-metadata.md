---
icon: kolena/properties-16
---

# :kolena-properties-20: Extracting Metadata from Images

This guide outlines how to configure the extraction of metadata from Images on Kolena. Follow the steps below
to get started with Automatic Metadata Extraction for Images.

## Configuring Metadata Extraction

??? "1. Navigate to Dataset Details"
    Scroll down to the "Details" page of your dataset.

    <figure markdown>
    ![Navigating to Configuration](../assets/images/navigate-to-text-extraction-config-dark.gif#only-dark)
    ![Navigating to Configuration](../assets/images/navigate-to-text-extraction-config-light.gif#only-light)
    <figcaption>Navigating to Metadata Configuration</figcaption>
    </figure>

??? "2. Select Image Fields and Properties"
    Identify and select the image field(s) from your dataset that you want to analyze.
    Also select the properties of the field(s) you wish to extract.

    In the examble below we extract properties from the `best_answer` and `question` fields. For the `best_answer` field,
    we display `word_count` and `topic_tag`, whereas for the `question` field we display `word_count`, `readability` and
    `question_type`.

    <figure markdown>
    ![Select Properties of Text Fields](../assets/images/select-text-extraction-properties-dark.gif#only-dark)
    ![Select Properties of Text Fields](../assets/images/select-text-extraction-properties-light.gif#only-light)
    <figcaption>Select Specific Properties of Interest For Relevant Fields</figcaption>
    </figure>

??? "3. Edit Metadata Configuration"
    To make additional metadata visible (or to hide existing metadata), the configuration can be edited.

    This will add/remove metadata properties. The example below shows how to add the `character_count` property
    to the `best_answer`. The properties shown in purple
    are the automatically extracted properties.

    <figure markdown>
    ![Edit Which Properties Are Visible](../assets/images/edit-text-extraction-properties-dark.gif#only-dark)
    ![Edit Which Properties Are Visible](../assets/images/edit-text-extraction-properties-light.gif#only-light)
    <figcaption>Example of adding `character_count` to the list of extracted properties</figcaption>
    </figure>

!!! example

    <figure markdown>
    ![Hydrated Question](../assets/images/hydrated-text-example-light.png#only-light)
    ![Hydrated Question](../assets/images/hydrated-text-example-dark.png#only-dark)
    <figcaption>Example of Hydrated Text - Note that the
        purple metadata indicate that they are auto-extracted </figcaption>
    </figure>

## Available Image Metadata Properties

The following properties are available for automatic image metadata extraction:

| Feature Name                  | Brief Description                              |
|-------------------------------|------------------------------------------------|
| [Aspect Ratio](#aspect-ratio) | Ratio of image width to height |
| [Brightness](#brightness) | Average pixel intensity |
| [Contrast](#contrast) | Standard deviation of pixel intensity |
| [Height](#height) | Height of image in pixels |
| [Pixel Entropy](#pixel-entropy) | Entropy of color distribution |
| [Sharpness](#sharpness) | Edge density from Canny edge detection |
| [Size](#size) | Product of image height and width |
| [Symmetry](#symmetry) | Level of horizontal symmetry in image |
| [Width](#width) | Width of image in pixels |

## Feature Descriptions

### Aspect Ratio

**Aspect ratio** measures the ratio of the image width to its height. It can be useful in scenarios
where the image shape or dimensions impact the analysis or model performance.

$$
\text{Aspect Ratio} = \frac{\text{Width}}{\text{Height}}
$$

!!! example

    "What phenomenon was conclusively proven by J. B. Rhine?" has  **55** characters (including spaces).

### Brightness

**Brightness** measures the average pixel intensity of an image, which can indicate how light or dark the image appears.
It can be useful in scenarios where the brightness impact the analysis or model performance.

$$
\text{Brightness} = \frac{\sum \text{Pixel Intensities}}{\text{Number of Pixels} \times \text{Max Pixel Intensity}}
$$

!!! example

    "Lindenstrauss" is considered a difficult word.

### Contrast

**Contrast** measures the standard deviation of pixel intensities in an image, indicating the degree of variation
between light and dark areas. The value is normalized by a maximum contrast value.

$$
\text{Contrast} = \min\left(1, \frac{\sigma_{\text{Pixel Intensities}}}{\text{Max Contrast}}\right)
$$

!!! example

    "Lindenstrauss" is considered a difficult word.

### Height

**Height** measures the height of the image in pixels. This is a straightforward dimension indicating the number
of pixel rows in the image.

$$
\text{Height} = \text{Number of Pixel Rows}
$$

!!! example

    "Lindenstrauss" is considered a difficult word.

### Pixel Entropy

**Pixel entropy** measures the entropy of the color distribution in an image, providing a measure of the image's
complexity or randomness. Higher entropy indicates more complexity.

$$
\text{Pixel Entropy} = -\sum_{i=1}^{N} p_i \log(p_i)
$$

!!! example

    "Lindenstrauss" is considered a difficult word.

where \( p_i \) is the probability of each unique pixel value.

### Sharpness

**Sharpness** measures the edge density in an image using the Canny edge detection algorithm. This can indicate how clear
or blurred an image is. The value is the proportion of edge pixels to total pixels.

$$
\text{Sharpness} = \frac{\text{Number of Edge Pixels}}{\text{Total Number of Pixels}}
$$

!!! example

    "Lindenstrauss" is considered a difficult word.

### Size

**Size** measures the product of the image height and width, giving the total number of pixels in the image.

$$
\text{Size} = \text{Height} \times \text{Width}
$$

!!! example

    "Lindenstrauss" is considered a difficult word.

### Symmetry

**Symmetry** measures the level of horizontal symmetry in an image by comparing the left and right halves.
The value ranges from 0 to 1, with 1 indicating perfect symmetry.

$$
\text{Symmetry} = 1 - \frac{\text{MSE(Left Half, Mirrored Right Half)}}{\text{Max Symmetry}}
$$

!!! example

    "Lindenstrauss" is considered a difficult word.

### Width

**Width** measures the width of the image in pixels. This is a straightforward dimension indicating the number of
pixel columns in the image.

$$
\text{Width} = \text{Number of Pixel Columns}
$$

!!! example

    "Lindenstrauss" is considered a difficult word.
