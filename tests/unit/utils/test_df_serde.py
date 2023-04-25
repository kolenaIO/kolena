import numpy as np

from kolena._utils.serde import deserialize_embedding_vector
from kolena._utils.serde import serialize_embedding_vector


def test_embedding_vector_serialization() -> None:
    dtypes = [str, np.int32, np.int64, np.complex128, "b", "B", ">H", "<f", "d", "i4", "u4", "f8", "c16", "a25", "U25"]
    for dtype in dtypes:
        want = np.array([[[1.0, 2.2], [3 + 1 / 3, 4]], [[5, 6], [7, 8]]]).astype(dtype)
        serialized = serialize_embedding_vector(want)
        got = deserialize_embedding_vector(serialized)
        assert np.array_equal(got, want)
        assert got.dtype == want.dtype
