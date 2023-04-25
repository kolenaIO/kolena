import numpy as np
import pandas as pd
import pandera as pa
import pytest

from kolena.fr.datatypes import ImageResultDataFrameSchema
from kolena.fr.datatypes import PairResultDataFrameSchema


def test_image_result_schema() -> None:
    def image_chip() -> np.ndarray:
        return (np.random.rand(112, 112, 3) * 256).astype(np.uint8)

    batch_size = 3
    df = pd.DataFrame(
        dict(
            image_id=list(range(batch_size)),
            bounding_box=[np.random.rand(4).astype(np.float32) for _ in range(batch_size)],
            landmarks_input_image=[image_chip() for _ in range(batch_size)],
            landmarks=[np.random.rand(10).astype(np.float32) for _ in range(batch_size)],
            quality_input_image=[image_chip() for _ in range(batch_size)],
            quality=np.random.rand(batch_size).astype(np.float64).tolist(),
            acceptability=np.random.rand(batch_size).astype(np.float64).tolist(),
            fr_input_image=[image_chip() for _ in range(batch_size)],
            embedding=[np.random.rand(256).astype(np.float32) for _ in range(batch_size)],
            failure_reason=[None] * batch_size,
        ),
    )
    ImageResultDataFrameSchema.validate(df)

    # should fail validation
    df.at[0, "bounding_box"] = "not an array"
    with pytest.raises(pa.errors.SchemaError):
        ImageResultDataFrameSchema.validate(df)

    # null values for nullable columns should not fail validation
    df.iloc[0] = (0, None, None, None, None, None, None, None, None, "whoopsies")
    ImageResultDataFrameSchema.validate(df)


def test_pair_result_schema() -> None:
    batch_size = 10
    df = pd.DataFrame(
        dict(
            image_pair_id=list(range(batch_size)),
            similarity=np.random.rand(batch_size).astype(np.float64).tolist(),
        ),
    )
    PairResultDataFrameSchema.validate(df)

    # null values for similarity (nullable) should not fail validation
    df.iloc[0] = (0, None)
    PairResultDataFrameSchema.validate(df)
