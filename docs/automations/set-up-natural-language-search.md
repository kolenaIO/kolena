---
icon: kolena/search-around-20
---

# :kolena-search-around-20: Setting up Natural Language Search

Kolena supports natural language and similar image search
across image data registered to the platform.
Users may set up this functionality by enabling the automated embedding extraction process
or manually extracting and uploading corresponding search embeddings using a Kolena provided package.

## Setting up Automated Embedding extraction

??? "Requirements"
    - This feature is currently supported for Amazon S3 integrations.
    - Kolena requires access to the content of your images.
    Read [Connecting Cloud Storage: Amazon S3](../connecting-cloud-storage/amazon-s3.md) for more details.
    - Only account administrators are able to change this setting.

Embedding extractions allow you to find datapoints using natural language or similarity between desired datapoints.
To enable automated embedding, navigate to "Organization Settings" available on your profile menu, top right of the screen.
Under the "Automations" tab, Enable the Automated Embeddings Extraction by Kolena option.

<figure markdown>
![Defining Metrics](../assets/images/automated-embeddings-extraction.gif)
<figcaption>Automated Embeddings Extraction</figcaption>
</figure>

Once this setting is enabled, embeddings for new and edited datapoints in your datasets will be automatically extracted.

## Uploading Custom Embeddings

If your organization restricts Kolena’s access to images, or if you use custom logic for embedding extraction,
 you can upload embeddings to enable Natural Language and Similar Image search on Kolena. For guidance,
 refer to the documentation on [Uploading Custom Embeddings](../dataset/advanced-usage/upload-embeddings.md).
