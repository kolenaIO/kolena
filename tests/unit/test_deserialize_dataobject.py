# Copyright 2021-2024 Kolena Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from kolena.io import _deserialize_dataobject


def test__deserialize_dataobject() -> None:
    bounding_box = {
        "top_left": [242.49460056994644, 636.9814293847943],
        "bottom_right": [1150.4028821200359, 920.0],
        "data_type": "ANNOTATION/BOUNDING_BOX",
        "label": "ego_vehicle",
    }

    image = {"locator": "s3://bucket/key", "data_type": "TEST_SAMPLE/IMAGE"}
    curve = {
        "curves": [{"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]}],
        "data_type": "PLOT/CURVE",
        "title": "test",
        "x_label": "x_label",
        "y_label": "y_label",
    }
    image_asset = {"locator": "s3://bucket/key", "data_type": "ASSET/IMAGE"}
    thresholded = {"threshold": 0.3, "data_type": "METRICS/THRESHOLDED"}

    deserialized_bounding_box = _deserialize_dataobject(bounding_box)
    deserialized_image = _deserialize_dataobject(image)
    deserialized_image_asset = _deserialize_dataobject(image_asset)
    deserialized_thresholded = _deserialize_dataobject(thresholded)
    deserialized_curve = _deserialize_dataobject(curve)
    from kolena.workflow.plot import Curve, CurvePlot
    from kolena.annotation import BoundingBox
    from kolena._experimental.workflow import ThresholdedMetrics
    from kolena.asset import ImageAsset
    from kolena.workflow.test_sample import Image

    assert deserialized_bounding_box == BoundingBox(
        top_left=[242.49460056994644, 636.9814293847943],
        bottom_right=[1150.4028821200359, 920.0],
        label="ego_vehicle",
    )
    assert deserialized_image == Image(locator="s3://bucket/key")
    assert deserialized_curve == CurvePlot(
        curves=[Curve(x=[1, 2, 3], y=[4, 5, 6])],
        title="test",
        x_label="x_label",
        y_label="y_label",
    )
    assert deserialized_image_asset == ImageAsset(locator="s3://bucket/key")
    assert deserialized_thresholded == ThresholdedMetrics(threshold=0.3)
