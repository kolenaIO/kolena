---
icon: kolena/media-20

---
# :kolena-media-20: Extracting Metadata from Images

This guide outlines how to configure the extraction of metadata from images on Kolena. Follow the steps below to
get started with Automatic Metadata Extraction for Images.

## Configuring Image Metadata Extraction

??? "1. Navigate to Dataset Details"
    Scroll down to the "Details" page of your dataset.

    <figure markdown>
    ![Navigating to Configuration](../../assets/images/image-metadata-open-config-dark.gif#only-dark)
    ![Navigating to Configuration](../../assets/images/image-metadata-open-config-light.gif#only-light)
    <figcaption>Navigating to Metadata Configuration</figcaption>
    </figure>

??? "2. Select Image Locators and Metadata"
    Identify and select locator column that stores references to your images
    from the dataset that you want to analyze.
    Also select the image properties you wish to extract.

    In the example below we extract properties from the `locator` which holds the location of images.
    For the `locator` field, we select all available properties.

    <figure markdown>
    ![Configure and run](../../assets/images/image-metadata-run-dark.gif#only-dark)
    ![Configure and run](../../assets/images/image-metadata-run-light.gif#only-light)
    <figcaption>Selecting and Executing Extraction</figcaption>
    </figure>

??? "3. Edit Metadata Configuration"
    To make additional metadata visible (or to hide existing metadata), the configuration can be edited.

    This will add/remove metadata metadata. The example below shows how to add or remove metadata from
    the `locator` files.

    <figure markdown>
    ![Edit Which Properties Are Visible](../../assets/images/edit-image-extraction-properties-dark.gif#only-dark)
    ![Edit Which Properties Are Visible](../../assets/images/edit-image-extraction-properties-light.gif#only-light)
    <figcaption>Example of editing the list of extracted metadata</figcaption>
    </figure>

!!! example
    Once the properties are extracted they will be presented on the dataset as image metadata. You can interact with
    these metadata similar to how interact with uploaded metadata. This means you are able to filter, sort, create test cases
    and create plots in the debugger tab with them.
    <figure markdown>
    ![Hydrated Question](../../assets/images/hydrated-image-example-light.png#only-light)
    ![Hydrated Question](../../assets/images/hydrated-image-example-dark.png#only-dark)
    <figcaption>Example of Hydrated Text - Note that the
        purple metadata indicate that they are auto-extracted </figcaption>
    </figure>

## Available Image Metadata Properties

The following properties are available for automatic image metadata extraction:

| Feature Name                  | Brief Description                              |
|-------------------------------|------------------------------------------------|
| Pixel Entropy | Entropy of the color distribution in the image |
| Sharpness | Edge density measured by Canny edge detection |
| Contrast | Standard deviation of pixel intensities |
| Brightness | Average of pixel intensities: overall lightness |
| Aspect Ratio | Ratio of image width to height |
| Height | The height of the image in pixels |
| Width | The width of the image in pixels |
| Size | Product of image height and width |
| Symmetry | Comparison with horizontally flipped version |
