---
icon: kolena/studio-16
---

# Fomatting data for Audio tasks

In this document we will review best practices when setting up Kolena datasets for Audio
problems.

## Basics

### Using the `locator`

Kolena uses references to files stored in your cloud storage to render them.
Refer to ["Connecting Cloud Storage"](../../../connecting-cloud-storage/)
for detials on how to configure this.

Audio samples can be visualized on Kolnena one of two ways.

**Gallary mode**: visulizes each audio as a tile.
To enable the Gallary view store references to audio files in a column named `locator`. `locator` can be used as
the unique identifier of the datapoint which is also refereced by your model results.

!!! tip
    You can use [`TimeSegment`](../../../reference/annotation.md#kolena.annotation.TimeSegment)'s
    or [`LabeledTimeSegment`](../../../reference/annotation.md#kolena.annotation.LabeledTimeSegment)
    to highlight segments of your audio file that are of interest to you.

!!! example
    The [:kolena-widget-16: Speaker Diarization ↗](https://github.com/kolenaIO/kolena/tree/trunk/examples/dataset/speaker_diarization)
    example showcases how audio files can be uploaded in Gallary mode.

<figure markdown>
![Audio Gallary View](../../../assets/images/gallary-audio-dark.png#only-dark)
![Audio Gallary View](../../../assets/images/gallary-audio-light.png#only-light)
<figcaption>Gallary View</figcaption>
</figure>

**Tabular**: visulizes each datapoint as a row on a table with the audio file as an asset evailable for replay. To enable
this view, please use [`AudioAsset`](../../../reference/asset.md#kolena.asset.AudioAsset) to link your audio files
to your dataset.

!!! example
    The [:kolena-widget-16: Automatic Speech Recognition ↗](https://github.com/kolenaIO/kolena/tree/trunk/examples/dataset/automatic_speech_recognition)
    example showcases how `AudioAsset`s can be attached to datapoints.

<figure markdown>
![Audio Gallary View](../../../assets/images/tabular-audio-dark.png#only-dark)
![Audio Gallary View](../../../assets/images/tabular-audio-light.png#only-light)
<figcaption>Tabular View</figcaption>
</figure>

Kolena supports `flac`, `mp3`, `wav`, `acc`, `ogg`, `ra` and other web browser supported audio types.

### Using fields

You can add additional informaiton about your audio files or other features of your data by
adding columns to the `.CSV` file with the meta-data name and values in each row.
Below is an example datapoint:

| Locator                                                | Num Speakers | Average Amplitude | audio_length |
|--------------------------------------------------------|--------------|------------------|--------------|
| `s3://kolena-public-examples/ICSI-corpus/data/audio/Bdb001/interval0.mp3` | 5            | 0.08951   | 448.454 |

## Uploading Model Results

Model results contian your model inferences as well as any custom metrics that you wish to monitor on Kolena.
The data structure of model resutls is very similar to the structure of a dataset.

* make sure to link your inferences to the dataset using the same unique ID you used when uploading the dataset.
* use [`ScoredTimeSegment`](../../../reference/annotation.md#kolena.annotation.ScoredTimeSegment) or [`ScoredLabeledTimeSegment`](../../../reference/annotation.md#kolena.annotation.ScoredLabeledTimeSegment)
 annotations to indicate the inference confidece score.
