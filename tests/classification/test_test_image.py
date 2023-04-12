import kolena.classification.metadata
from kolena.classification import TestImage


def test_image__serde() -> None:
    original = TestImage(
        locator="s3://test-bucket/path/to/file.png",
        dataset="test-dataset",
        labels=["one", "2", "$3", "^4", "!@#$%^&*()"],
        metadata={
            "none": None,
            "str": "string",
            "float": 1.0,
            "int": 500,
            "bool": False,
            "bbox": kolena.classification.metadata.BoundingBox(top_left=(0, 0), bottom_right=(100, 100)),
            "landmarks": kolena.classification.metadata.Landmarks(points=[(0, 0), (100, 100), (100, 0), (0, 0)]),
            "asset": kolena.classification.metadata.Asset(locator="gs://gs-bucket/path/to/asset.jpg"),
        },
    )

    df = kolena.classification.test_case.TestCase._to_data_frame([original])
    assert len(df) == 1

    recovered = [TestImage._from_record(record) for record in df.itertuples()][0]
    assert original == recovered
