---
icon: kolena/widget-16
status: new
---

# :kolena-widget-20: Object Detection (2D)

!!! warning "Legacy Warning"

    Content in this section reflects **outdated practices** or **deprecated features**. It's recommended to avoid using these in new developments.

    While existing implementations using these features will continue to receive support, we strongly advise adopting the latest standards and tools for new projects to ensure optimal performance and compatibility. For more information and up-to-date practices, please refer to our newest documentation at [docs.kolena.io](https://docs.kolena.io).



Object Detection (OD) is a computer vision task that aims to classify and locate objects of interest presented in an
image. So, it can be viewed as a combination of localization and classification tasks.

This pre-built workflow is prepared for a 2D Object Detection problem and here is an example of using this workflow
on the COCO dataset.

<div class="grid cards" markdown>
- [:kolena-widget-20: Example: Object Detection (2D) ↗](https://github.com/kolenaIO/kolena/tree/hotfix/0.100.x/examples/object_detection_2d)

    ![Example 2D bounding boxes from the COCO dataset.](../../assets/images/COCO-transportation.jpeg)

    ---

    2D Object Detection using the [COCO](https://cocodataset.org/#overview) dataset
</div>

::: kolena._experimental.object_detection
