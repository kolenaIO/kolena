"""
Metadata associated with a :class:`kolena.detection.TestImage`.

.. code-block:: python

    test_image = TestImage("s3://bucket/path/to/image.png", metadata=dict(
        input_landmarks=Landmarks(*landmarks),
        input_bounding_box=BoundingBox(*bounding_box),
        image_grayscale=Asset("s3://bucket/path/to/image_grayscale.png"),
    ))
"""
from kolena.detection._internal.metadata import Annotation
from kolena.detection._internal.metadata import Asset
from kolena.detection._internal.metadata import BoundingBox
from kolena.detection._internal.metadata import Landmarks
from kolena.detection._internal.metadata import MetadataElement

__all__ = [
    "Annotation",
    "BoundingBox",
    "Landmarks",
    "Asset",
    "MetadataElement",
]
