---
icon: kolena/properties-16
---

# :kolena-properties-20: Automatically Extract Bounding Box Properties

This guide outlines how to configure the extraction of properties from bounding box fields on Kolena.

## Configuring Bounding Box Property Extraction

Wait FE implementation

## Available Bounding Box Properties
The following properties are available for automatic bounding box property extraction, there are two types of properties:
1. Bounding Box Property: the property related to a single bounding box
2. Bounding Box Group Property: the property related to the group of bounding boxes contained in a bounding box field

| Feature Name                                                                      | Brief Description                                                                                    | Property Type              |
|-----------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------|----------------------------|
| [Aspect Ratio](#aspect-ratio)                                                     | The Aspect ratio of the bounding box width / height                                                  | Bounding Box Property |
| [Relative Area](#relative-area)                                                   | The relative area of the bounding box with respect to the image                                      | Bounding Box Property |
| [Relative Distance To Image Centre](#relative-distance-to-image-centre)           | The Euclidean distance between the bounding box's center and the center of the image                 | Bounding Box Property |
| [Object Count](#object-count)                                                     | Count the number of bounding boxes in the group                                                      | Bounding Box Group Property |
| [Object Type Count](#object-type-count)                                           | Count the types of objects in the group according to label field                                     | Bounding Box Group Property |
| [Combined Area Ratio](#combined-area-ratio)                                       | Sum of the bounding box size in the group                                                            | Bounding Box Group Property |
| [Bbox Intersection](#bbox-interection)                                            | Sum of the IOU of all bounding box pairs in the group                                                | Bounding Box Group Property |
| [Entropy Of Distribution](#entropy-of-distribution)                               | Entropy of the distribution of bounding boxes within a 3 by 3 grid                                   | Bounding Box Group Property |
| [Mean Aspect Ratio](#mean-aspect-ratio)                                           | The mean of the aspect ratios of a group of bounding boxes                                           | Bounding Box Group Property |
| [Mean Distance Between Bboxes](#mean-distance-between-bboxes)                     | The mean of the Euclidean distance between each pair of bounding box in the group                    | Bounding Box Group Property |
| [Mean Relative Area](#mean-relative-area)                                         | The mean of the relative area of bounding boxes in the group                                         | Bounding Box Group Property |
| [Mean Relative Distance To Image Centre](#mean-relative-distance-to-image-centre) | The mean of the relative distance between each of the bounding box and the image center in the group | Bounding Box Group Property |
| [Std Aspect Ratio](#std-aspect-ratio)                                             | The standard deviation of the aspect ratios of a group of bounding boxes                             | Bounding Box Group Property |
| [Std Distance Between Bboxes](#std-distance-between-bboxes)                      | The standard deviation  of the Euclidean distance between each pair of bounding box in the group     | Bounding Box Group Property |
| [Std Relative Area](#std-relative-area)                                          | The standard deviation of the relative area of bounding boxes in the group                                         | Bounding Box Group Property |
| [Std Relative Distance To Image Centre](#std-relative-distance-to-image-centre)  | The standard deviation of the relative distance between each of the bounding box and the image center in the group | Bounding Box Group Property |
