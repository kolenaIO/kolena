# Copyright 2021-2023 Kolena Inc.
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
import json
from typing import Any
from typing import cast
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Type

import numpy as np
import pandas as pd
import pandera as pa
from pandera.extensions import register_check_method
from pandera.typing import Series

from kolena._utils.asset_path_mapper import AssetPathMapper
from kolena._utils.dataframes.validators import validate_df_schema
from kolena._utils.datatypes import LoadableDataFrame
from kolena._utils.serde import deserialize_embedding_vector
from kolena._utils.serde import serialize_embedding_vector
from kolena._utils.serde import with_serialized_columns


__ALLOWED_INTEGRAL_DTYPES = {
    int,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    np.int8,
    np.int32,
    np.int64,
    np.dtype("uint8"),
    np.dtype("uint16"),
    np.dtype("uint32"),
    np.dtype("uint64"),
    np.dtype("int8"),
    np.dtype("int32"),
    np.dtype("int64"),
}
__ALLOWED_NUMERIC_DTYPES = {
    *__ALLOWED_INTEGRAL_DTYPES,
    float,
    np.float16,
    np.float32,
    np.float64,
    np.longdouble,
    np.dtype("float16"),
    np.dtype("float32"),
    np.dtype("float64"),
    np.dtype("longdouble"),
}


# NOTE: these are actually np.ndarrays, but are declared as objects for compatibility with pandera schemas (as far as
#  pandas is concerned these cells are type 'object')
BoundingBox = object  # bounding box [top_left_x, top_left_y, bottom_right_x, bottom_right_y]
Landmarks = object  # landmarks [left_eye_{x,y}, right_eye_{x,y}, nose_{x,y}, left_mouth_{x,y}, right_mouth_{x,y}]
RGBImage = object  # MxNx3 uint8 ndarray representing an RGB image of width N and height M
EmbeddingVector = object  # embedding vector (arbitrary size and data type)
JSONObject = object


# note that all checks accept nulls -- rely on pandera to properly enforce nullability
def _validate_ndarray(series: Series) -> bool:
    return series.dropna().apply(lambda cell: isinstance(cell, np.ndarray)).all()


@register_check_method()
def _validate_bounding_box(series: Series) -> bool:
    def validate_cell(cell: np.ndarray) -> bool:
        return cell.dtype in __ALLOWED_NUMERIC_DTYPES and cell.shape == (4,)

    return _validate_ndarray(series) and series.dropna().apply(validate_cell).all()


@register_check_method()
def _validate_landmarks(series: Series) -> bool:
    def validate_cell(cell: np.ndarray) -> bool:
        return cell.dtype in __ALLOWED_NUMERIC_DTYPES and cell.shape == (10,)

    return _validate_ndarray(series) and series.dropna().apply(validate_cell).all()


@register_check_method()
def _validate_rgb_image(series: Series) -> bool:
    def validate_cell(cell: np.ndarray) -> bool:
        return cell.dtype == np.uint8 and len(cell.shape) == 3 and cell.shape[2] == 3

    return _validate_ndarray(series) and series.dropna().apply(validate_cell).all()


@register_check_method()
def _validate_embedding_vector(series: Series) -> bool:
    return _validate_ndarray(series)


@register_check_method()
def _validate_optional_dimension(series: Series) -> bool:
    # treat -1 as the value for "not specified"
    return (
        series.dropna().apply(lambda cell: type(cell) in __ALLOWED_INTEGRAL_DTYPES and (cell > 0 or cell == -1)).all()
    )


@register_check_method()
def _validate_json_object(series: Series) -> bool:
    # TODO: more detailed validation? validate that each cell can serialize to JSON
    return series.dropna().apply(lambda cell: isinstance(cell, dict)).all()


@register_check_method()
def _validate_tags(series: Series) -> bool:
    def is_dict_str_str(cell: dict) -> bool:
        return all(isinstance(k, str) and isinstance(v, str) for k, v in cell.items())

    return series.dropna().apply(lambda cell: isinstance(cell, dict) and is_dict_str_str(cell)).all()


def _as_json(value: Optional[Any]) -> Optional[str]:
    return json.dumps(value) if value is not None else None


def _from_json(value: Optional[str]) -> Optional[Any]:
    return json.loads(value) if value is not None else None


class TestImageDataFrameSchema(pa.SchemaModel):
    # internal ID corresponding to this image
    image_id: Series[pa.typing.Int64] = pa.Field(coerce=True)

    # external locator pointing to image in bucket
    locator: Series[pa.typing.String] = pa.Field(coerce=True, _validate_locator=())

    # specify source dataset, e.g. "CIFAR-10"
    data_source: Series[pa.typing.String] = pa.Field(coerce=True, nullable=True)

    # -1 used to specify "unspecified"
    width: Series[pa.typing.Int64] = pa.Field(coerce=True, _validate_optional_dimension=())
    height: Series[pa.typing.Int64] = pa.Field(coerce=True, _validate_optional_dimension=())

    # specify that this image is an augmented version of another (registered) image
    original_locator: Series[pa.typing.String] = pa.Field(coerce=True, nullable=True, _validate_locator=())

    # free-form specification describing the augmentation applied to the image, e.g. {"rotate": 90}
    # should not be specified unless original_locator is present
    augmentation_spec: Series[JSONObject] = pa.Field(coerce=True, nullable=True, _validate_json_object=())

    # ground truth bounding box -- if absent, no ground truth will be available for display
    bounding_box: Series[BoundingBox] = pa.Field(coerce=True, nullable=True, _validate_bounding_box=())

    # ground truth landmarks -- if absent, no ground truth will be available for display
    landmarks: Series[Landmarks] = pa.Field(coerce=True, nullable=True, _validate_landmarks=())

    # specify a set of tags to apply to this object in the form {category: value}
    # note that this format intentionally restricts tags to a single value per category
    tags: Series[JSONObject] = pa.Field(coerce=True, _validate_tags=())


TestImageRecord = Tuple[
    str,
    Optional[str],
    int,
    int,
    Optional[str],
    Optional[Dict[str, Any]],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Dict[str, Any],
]
TEST_IMAGE_COLUMNS = [
    "locator",
    "data_source",
    "width",
    "height",
    "original_locator",
    "augmentation_spec",
    "bounding_box",
    "landmarks",
    "tags",
]


class TestCaseDataFrameSchema(pa.SchemaModel):
    locator_a: Series[pa.typing.String] = pa.Field(coerce=True, _validate_locator=())
    locator_b: Series[pa.typing.String] = pa.Field(coerce=True, _validate_locator=())
    is_same: Series[pa.typing.Bool] = pa.Field(coerce=True)


TestCaseRecord = Tuple[str, str, bool]
TEST_CASE_COLUMNS = ["locator_a", "locator_b", "is_same"]


class ImageDataFrameSchema(pa.SchemaModel):
    image_id: Series[pa.typing.Int64] = pa.Field(coerce=True)
    locator: Series[pa.typing.String] = pa.Field(coerce=True)


class ImageResultDataFrameSchema(pa.SchemaModel):
    #: [**Required**] The ID of the image corresponding to this record.
    image_id: Series[pa.typing.Int64] = pa.Field(coerce=True)

    #: [Optional] A bounding box around the face detected in this image.
    bounding_box: Optional[Series[BoundingBox]] = pa.Field(coerce=True, nullable=True, _validate_bounding_box=())

    #: [Optional] RGB image (``np.ndarray`` with cells of type ``np.uint8``) corresponding to the input to the
    #: "landmarks" model in the face recognition pipeline.
    landmarks_input_image: Optional[Series[RGBImage]] = pa.Field(coerce=True, nullable=True, _validate_rgb_image=())

    #: [Optional] A 10-element array with ``(x, y)`` coordinates corresponding to the left eye, right eye, nose tip,
    #: left mouth corner, right mouth corner of the detected face.
    landmarks: Optional[Series[Landmarks]] = pa.Field(coerce=True, nullable=True, _validate_landmarks=())

    #: [Optional] RGB image (``np.ndarray`` with cells of type ``np.uint8``) corresponding to the input to the "quality"
    #: model in the face recognition pipeline.
    quality_input_image: Optional[Series[RGBImage]] = pa.Field(coerce=True, nullable=True, _validate_rgb_image=())

    #: [Optional] Score produced by the "quality" model in the face recognition pipeline.
    quality: Optional[Series[pa.typing.Float64]] = pa.Field(coerce=True, nullable=True)

    #: [Optional] Score produced by the "acceptability" model in the face recognition pipeline.
    acceptability: Optional[Series[pa.typing.Float64]] = pa.Field(coerce=True, nullable=True)

    #: [Optional] RGB image (``np.ndarray`` with cells of type ``np.uint8``) corresponding to the input to the facial
    #: embeddings extraction model in the face recognition pipeline.
    fr_input_image: Optional[Series[RGBImage]] = pa.Field(coerce=True, nullable=True, _validate_rgb_image=())

    #: [Optional] Embedding vector (``np.ndarray``) extracted representing the face detected in the image. An empty cell
    #: (no array is provided) indicates a failure to enroll for the image, i.e. no face detected.
    embedding: Series[EmbeddingVector] = pa.Field(coerce=True, nullable=True, _validate_embedding_vector=())

    #: [Optional] The reason why the image was a failure to enroll.
    failure_reason: Optional[Series[pa.typing.String]] = pa.Field(coerce=True, nullable=True)


class _ResultStageFrameSchema(pa.SchemaModel):
    test_run_id: Series[pa.typing.Int64] = pa.Field(coerce=True)
    image_id: Series[pa.typing.Int64] = pa.Field(coerce=True)
    # null values for the following columns indicate a failure to enroll
    bbox: Series[pa.typing.String] = pa.Field(coerce=True, nullable=True)
    lmks: Series[pa.typing.String] = pa.Field(coerce=True, nullable=True)
    embedding: Series[pa.typing.String] = pa.Field(coerce=True, nullable=True)
    assets: Series[pa.typing.String] = pa.Field(coerce=True)  # should be {} when no data is present
    metadata: Series[pa.typing.String] = pa.Field(coerce=True)  # should be {} when no data is present
    failure_reason: Series[pa.typing.String] = pa.Field(coerce=True, nullable=True)  # should be null unless failure


class _ImageChipsDataFrameSchema(pa.SchemaModel):
    test_run_id: Series[pa.typing.Int64] = pa.Field(coerce=True)
    image_id: Series[pa.typing.Int64] = pa.Field(coerce=True)
    key: Series[pa.typing.String] = pa.Field(coerce=True)
    uuid: Series[pa.typing.String] = pa.Field(coerce=True)
    image: Series[RGBImage] = pa.Field(coerce=True, _validate_rgb_image=())


class EmbeddingDataFrameSchema(pa.SchemaModel):
    image_id: Series[pa.typing.Int64] = pa.Field(coerce=True)

    #: The extracted embedding(s) corresponding to the ``image_id``. A missing embedding indicates a failure to enroll
    #: for the image.
    #:
    #: For images with only one extracted embedding, the ``embedding`` is a one-dimensional ``np.ndarray`` with length
    #: matching the length of the extracted embedding. When multiple embeddings were extracted from a single image, the
    #: first dimension represents the index of the extracted embedding. For example, for an image with 3 extracted
    #: embeddings and embeddings of length 256, the ``embedding`` is an array of shape (3, 256).
    embedding: Series[EmbeddingVector] = pa.Field(nullable=True, coerce=True, _validate_embedding_vector=())


class PairDataFrameSchema(pa.SchemaModel):
    image_pair_id: Series[pa.typing.Int64] = pa.Field(coerce=True)
    image_a_id: Series[pa.typing.Int64] = pa.Field(coerce=True)
    image_b_id: Series[pa.typing.Int64] = pa.Field(coerce=True)


class PairResultDataFrameSchema(pa.SchemaModel):
    #: [**Required**] The ID of the image corresponding to this record.
    image_pair_id: Series[pa.typing.Int64] = pa.Field(coerce=True)

    #: [Optional] The similarity score computed between the two embeddings in this image pair. Should be left empty when
    #: either image in the pair is a failure to enroll.
    similarity: Series[pa.typing.Float64] = pa.Field(nullable=True, coerce=True)

    #: [Optional] Index of the embedding in ``image_a`` corresponding to this similarity score. Required when multiple
    #: embeddings are extracted per image and multiple similarity scores are computed per image pair.
    embedding_a_index: Optional[Series[pa.typing.Int64]] = pa.Field(nullable=True, coerce=True)

    #: [Optional] Index of the embedding in ``image_b`` corresponding to this similarity score. Required when multiple
    #: embeddings are extracted per image and multiple similarity scores are computed per image pair.
    embedding_b_index: Optional[Series[pa.typing.Int64]] = pa.Field(nullable=True, coerce=True)


class LoadedPairResultDataFrameSchema(TestCaseDataFrameSchema, PairResultDataFrameSchema):  # note inheritance
    image_a_fte: Series[pa.typing.Bool] = pa.Field(coerce=True)  # if image_a failed to enroll (FTE)
    image_b_fte: Series[pa.typing.Bool] = pa.Field(coerce=True)  # if image_b failed to enroll (FTE)


class TestImageDataFrame(pa.typing.DataFrame[TestImageDataFrameSchema], LoadableDataFrame["TestImageDataFrame"]):
    def as_serializable(self) -> pd.DataFrame:
        object_columns = ["augmentation_spec", "bounding_box", "landmarks", "tags"]
        return with_serialized_columns(self, object_columns)

    @classmethod
    def get_schema(cls) -> Type[TestImageDataFrameSchema]:
        return TestImageDataFrameSchema

    @classmethod
    def from_serializable(cls, df: pd.DataFrame) -> "TestImageDataFrame":
        object_columns = ["augmentation_spec", "bounding_box", "landmarks", "tags"]
        for col in object_columns:
            df[col] = df[col].apply(_from_json)
        ndarray_columns = ["bounding_box", "landmarks"]
        for col in ndarray_columns:
            df[col] = df[col].apply(lambda v: None if v is None else np.array(v))
        return cast(TestImageDataFrame, validate_df_schema(df, TestImageDataFrameSchema, trusted=True))


class TestCaseDataFrame(pa.typing.DataFrame[TestCaseDataFrameSchema], LoadableDataFrame["TestCaseDataFrame"]):
    @classmethod
    def get_schema(cls) -> Type[TestCaseDataFrameSchema]:
        return TestCaseDataFrameSchema


class LoadedPairResultDataFrame(
    pa.typing.DataFrame[LoadedPairResultDataFrameSchema],
    LoadableDataFrame["LoadedPairResultDataFrame"],
):
    @classmethod
    def get_schema(cls) -> Type[LoadedPairResultDataFrameSchema]:
        return LoadedPairResultDataFrameSchema


class ImageResultDataFrame(pa.typing.DataFrame[ImageResultDataFrameSchema]):
    ...


class _ResultStageFrame(pa.typing.DataFrame[_ResultStageFrameSchema]):
    @classmethod
    def from_image_result_data_frame(
        cls,
        test_run_id: int,
        load_uuid: str,
        df: ImageResultDataFrame,
        path_mapper: AssetPathMapper,
    ) -> "_ResultStageFrame":
        df_stage = pd.DataFrame(
            dict(
                test_run_id=test_run_id,
                image_id=df["image_id"],
                bbox=[
                    json.dumps(record.bounding_box.tolist())
                    if "bounding_box" in df.columns and record.bounding_box is not None
                    else None
                    for record in df.itertuples()
                ],
                lmks=[
                    json.dumps(record.landmarks.tolist())
                    if "landmarks" in df.columns and record.landmarks is not None
                    else None
                    for record in df.itertuples()
                ],
                embedding=[
                    serialize_embedding_vector(record.embedding) if record.embedding is not None else None
                    for record in df.itertuples()
                ],
                assets=[
                    json.dumps(
                        {
                            f"{key}_input_image": path_mapper.absolute_locator(
                                test_run_id=test_run_id,
                                load_uuid=load_uuid,
                                image_id=getattr(record, "image_id"),
                                key=key,
                            )
                            for key in ["landmarks", "quality", "fr"]
                            if f"{key}_input_image" in df.columns and getattr(record, f"{key}_input_image") is not None
                        },
                    )
                    for record in df.itertuples()
                ],
                metadata=[
                    json.dumps(
                        {
                            key: getattr(record, key)
                            for key in ["quality", "acceptability"]
                            # this works because both quality and acceptability are float types
                            if key in df.columns and not pd.isna(getattr(record, key))
                        },
                    )
                    for record in df.itertuples()
                ],
                failure_reason=df["failure_reason"] if "failure_reason" in df.columns else [None] * len(df),
            ),
        )
        return cast(_ResultStageFrame, validate_df_schema(df_stage, _ResultStageFrameSchema, trusted=True))


class _ImageChipsDataFrame(pa.typing.DataFrame[_ImageChipsDataFrameSchema]):
    @classmethod
    def from_image_result_data_frame(
        cls,
        test_run_id: int,
        load_uuid: str,
        df: ImageResultDataFrame,
    ) -> "_ImageChipsDataFrame":
        keys = ["landmarks", "quality", "fr"]
        df_chips_locators: List[pd.DataFrame] = []
        for key in keys:
            image_col = f"{key}_input_image"
            if image_col not in df.columns:
                columns = ["image_id", "image", "test_run_id", "uuid", "key"]
                df_chips_locators.append(pd.DataFrame([], columns=columns))
                continue
            df_key = df[["image_id", image_col]]
            df_key = df_key[df_key[image_col].notnull()]
            df_key["test_run_id"] = test_run_id
            df_key["uuid"] = load_uuid
            df_key["key"] = key
            df_key = df_key.rename(columns={image_col: "image"})
            df_chips_locators.append(df_key)
        df_merged = pd.concat(df_chips_locators)
        return cls(validate_df_schema(df_merged, _ImageChipsDataFrameSchema, trusted=True))


class ImageDataFrame(pa.typing.DataFrame[ImageDataFrameSchema], LoadableDataFrame["ImageDataFrame"]):
    @classmethod
    def get_schema(cls) -> Type[ImageDataFrameSchema]:
        return ImageDataFrameSchema


class EmbeddingDataFrame(pa.typing.DataFrame[EmbeddingDataFrameSchema], LoadableDataFrame["EmbeddingDataFrame"]):
    @classmethod
    def get_schema(cls) -> Type[EmbeddingDataFrameSchema]:
        return EmbeddingDataFrameSchema

    @classmethod
    def from_serializable(cls, df: pd.DataFrame) -> "EmbeddingDataFrame":
        records: List[Tuple[int, Optional[np.ndarray]]] = []
        for image_id, df_embedding in df.groupby("image_id"):
            image_id_int = int(image_id)  # type: ignore
            embeddings = [  # prevented from including both embeddings and None (FTE) during ingest
                deserialize_embedding_vector(record.embedding)
                for record in df_embedding.itertuples()
                if record.embedding is not None
            ]
            if len(embeddings) == 0:
                records.append((image_id_int, None))
            elif len(embeddings) == 1:
                records.append((image_id_int, embeddings[0]))
            else:
                embeddings_arr = np.concatenate([embedding.reshape(1, len(embedding)) for embedding in embeddings])
                records.append((image_id_int, embeddings_arr))
        df_stage = pd.DataFrame(records, columns=["image_id", "embedding"])
        return cast(EmbeddingDataFrame, validate_df_schema(df_stage, EmbeddingDataFrameSchema, trusted=True))


class PairDataFrame(pa.typing.DataFrame[PairDataFrameSchema], LoadableDataFrame["PairDataFrame"]):
    @classmethod
    def get_schema(cls) -> Type[PairDataFrameSchema]:
        return PairDataFrameSchema


PairResultDataFrame = pa.typing.DataFrame[PairResultDataFrameSchema]
