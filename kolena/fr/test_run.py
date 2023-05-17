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
import dataclasses
import json
from abc import ABC
from typing import List
from typing import Tuple

from deprecation import deprecated
from pydantic import validate_arguments
from pydantic.dataclasses import dataclass
from tqdm import tqdm

from kolena._api.v1.batched_load import BatchedLoad as LoadAPI
from kolena._api.v1.fr import Asset as AssetAPI
from kolena._api.v1.fr import TestRun as API
from kolena._utils import krequests
from kolena._utils import log
from kolena._utils.asset_path_mapper import AssetPathMapper
from kolena._utils.batched_load import _BatchedLoader
from kolena._utils.batched_load import init_upload
from kolena._utils.batched_load import upload_data_frame
from kolena._utils.consts import BatchSize
from kolena._utils.dataframes.validators import validate_df_record_count
from kolena._utils.dataframes.validators import validate_df_schema
from kolena._utils.frozen import Frozen
from kolena._utils.instrumentation import report_crash
from kolena._utils.instrumentation import WithTelemetry
from kolena._utils.serde import from_dict
from kolena._utils.validators import ValidatorConfig
from kolena.errors import InputValidationError
from kolena.fr import InferenceModel
from kolena.fr import Model
from kolena.fr import TestSuite
from kolena.fr._utils import upload_image_chips
from kolena.fr.datatypes import _ImageChipsDataFrame
from kolena.fr.datatypes import _ResultStageFrame
from kolena.fr.datatypes import EmbeddingDataFrame
from kolena.fr.datatypes import ImageDataFrame
from kolena.fr.datatypes import ImageResultDataFrame
from kolena.fr.datatypes import ImageResultDataFrameSchema
from kolena.fr.datatypes import PairDataFrame
from kolena.fr.datatypes import PairResultDataFrame
from kolena.fr.datatypes import PairResultDataFrameSchema


class TestRun(ABC, Frozen, WithTelemetry):
    """
    Interface to run tests for a :class:`kolena.fr.Model` on a set of
    :class:`kolena.fr.TestSuite` suites. Any in-progress tests for this model on these suites are resumed.

    For a streamlined interface, see :meth:`kolena.fr.test`.

    :param model: the model being tested.
    :param test_suite: the test suite on which to test the model.
    :param reset: overwrites existing inferences if set.
    """

    _id: int

    @dataclass(frozen=True, config=ValidatorConfig)
    class Data:
        id: int
        model: Model.Data
        test_suites: List[TestSuite.Data]

    def __init__(self, model: Model, test_suite: TestSuite, reset: bool = False):
        if reset:
            log.warn("overwriting existing inferences from this model (reset=True)")
        else:
            log.info("not overwriting any existing inferences from this model (reset=False)")

        request = API.CreateOrRetrieveRequest(model_id=model.data.id, test_suite_ids=[test_suite._id], reset=reset)
        res = krequests.post(
            endpoint_path=API.Path.CREATE_OR_RETRIEVE.value,
            data=json.dumps(dataclasses.asdict(request)),
        )
        krequests.raise_for_status(res)
        response = from_dict(data_class=TestRun.Data, data=res.json())

        self.data = response
        self._id = response.id
        self._model = model
        self._test_suite = test_suite
        self._reset = reset
        self._freeze()

    @classmethod
    @deprecated(details="use initializer :class:`kolena.fr.TestRun` directly", deprecated_in="0.58.0")
    def create_or_retrieve(cls, model: Model, test_suite: TestSuite, reset: bool = False) -> "TestRun":
        """
        Create a new test run for the provided :class:`kolena.fr.Model` on the provided :class:`kolena.fr.TestSuite`.
        If a test run for this model on this suite already exists, it is returned.

        :param model: the model being tested.
        :param test_suite: the test suite on which to test the model.
        :param reset: overwrites existing inferences if set.
        :return: the created or retrieved test run.
        """
        return TestRun(model, test_suite, reset=reset)

    @validate_arguments
    def load_remaining_images(self, batch_size: int = 10_000_000) -> ImageDataFrame:
        """
        Load a DataFrame containing records for each of the images in the configured test suite that does not yet have
        results from the configured model.

        :param batch_size: optionally specify the maximum number of image records to return. Defaults to ``10_000_000``.
        :return: DataFrame containing records for each of the images that must be processed.
        :raises InputValidationError: if the requested ``batch_size`` failed validation.
        :raises RemoteError: if images could not be loaded for any reason.
        """
        if batch_size <= 0:
            raise InputValidationError(f"invalid batch_size '{batch_size}': expected positive integer")
        log.info("loading remaining images for test run")
        init_request = API.InitLoadRemainingImagesRequest(
            test_run_id=self.data.id,
            batch_size=batch_size,
            load_all=self._reset,
        )
        with krequests.put(
            endpoint_path=API.Path.INIT_LOAD_REMAINING_IMAGES.value,
            data=json.dumps(dataclasses.asdict(init_request)),
            stream=True,
        ) as init_res:
            krequests.raise_for_status(init_res)

            load_uuid = None
            try:
                dfs: List[ImageDataFrame] = []
                for line in init_res.iter_lines():
                    partial_response = from_dict(data_class=LoadAPI.InitDownloadPartialResponse, data=json.loads(line))
                    load_uuid = partial_response.uuid
                    dfs.append(_BatchedLoader.load_path(partial_response.path, ImageDataFrame))
                log.info("loaded remaining images for test run")
                return _BatchedLoader.concat(dfs, ImageDataFrame)
            finally:
                _BatchedLoader.complete_load(load_uuid)

    def upload_image_results(self, df_image_result: ImageResultDataFrame) -> int:
        """
        Upload inference results for a batch of images.

        All columns except for ``image_id`` and ``embedding`` are optional. An empty ``embedding`` cell in a record
        indicates a failure to enroll. The ``failure_reason`` column can optionally be specified for failures to enroll.

        To provide more than one embedding extracted from a given image, include multiple records with the same
        ``image_id`` in ``df_image_result`` (one for each embedding extracted). Records for a given ``image_id`` must
        be submitted in the same ``df_image_result`` DataFrame, and **not** across multiple calls to
        ``upload_image_results``.

        :param df_image_result: DataFrame of any size containing records describing inference results for an image.
        :return: number of records successfully uploaded.
        :raises TypeValidationError: if the DataFrame failed type validation.
        :raises RemoteError: if the DataFrame was unable to be successfully ingested for any reason.
        """
        log.info("uploading inference results for test run")
        init_response = init_upload()

        asset_config_res = krequests.get(endpoint_path=AssetAPI.Path.CONFIG.value)
        krequests.raise_for_status(asset_config_res)
        asset_config = from_dict(data_class=AssetAPI.Config, data=asset_config_res.json())
        asset_path_mapper = AssetPathMapper(asset_config)

        df_validated = ImageResultDataFrame(validate_df_schema(df_image_result, ImageResultDataFrameSchema))
        validate_df_record_count(df_validated)
        df_image_chips = _ImageChipsDataFrame.from_image_result_data_frame(
            test_run_id=self.data.id,
            load_uuid=init_response.uuid,
            df=df_validated,
        )
        upload_image_chips(df_image_chips)
        df_result_stage = _ResultStageFrame.from_image_result_data_frame(
            test_run_id=self.data.id,
            load_uuid=init_response.uuid,
            df=df_validated,
            path_mapper=asset_path_mapper,
        )
        upload_data_frame(df_result_stage, BatchSize.UPLOAD_RECORDS.value, init_response.uuid)

        request = API.UploadImageResultsRequest(uuid=init_response.uuid, test_run_id=self.data.id, reset=self._reset)
        finalize_res = krequests.put(
            endpoint_path=API.Path.COMPLETE_UPLOAD_IMAGE_RESULTS.value,
            data=json.dumps(dataclasses.asdict(request)),
        )
        krequests.raise_for_status(finalize_res)
        response = from_dict(data_class=API.UploadImageResultsResponse, data=finalize_res.json())
        log.success("uploaded inference results for test run")
        return response.n_uploaded

    @validate_arguments
    def load_remaining_pairs(self, batch_size: int = 10_000_000) -> Tuple[EmbeddingDataFrame, PairDataFrame]:
        """
        Load DataFrames containing computed embeddings and records for each of the image pairs in the configured test
        suite that have not yet had similarity scores computed.

        This method should not be called until all images in the :class:`TestRun` have been processed.

        :param batch_size: optionally specify the maximum number of image pair records to return. Defaults to
            ``10_000_000``.
        :return: two DataFrames, one containing embeddings computed in the previous step (``df_embedding``) and one
            containing records for each of the image pairs that must be computed (``df_pair``). See documentation on
            :class:`kolena.fr.datatypes.EmbeddingDataFrameSchema` for expected format when multiple embeddings were
            uploaded from a single image in :meth:`kolena.fr.TestRun.upload_image_results`.
        :raises InputValidationError: if the requested ``batch_size`` failed validation.
        :raises RemoteError: if pairs could not be loaded for any reason.
        """
        if batch_size <= 0:
            raise InputValidationError(f"invalid batch_size '{batch_size}': expected positive integer")

        log.info("loading batch of image pairs for test run")
        init_request = API.InitLoadRemainingPairsRequest(
            test_run_id=self.data.id,
            batch_size=batch_size,
            load_all=self._reset,
        )
        with krequests.put(
            endpoint_path=API.Path.INIT_LOAD_REMAINING_PAIRS.value,
            data=json.dumps(dataclasses.asdict(init_request)),
            stream=True,
        ) as init_res:
            krequests.raise_for_status(init_res)

            load_uuid_embedding = None
            load_uuid_pair = None
            try:
                dfs_embedding = []
                dfs_pair = []
                for line in init_res.iter_lines():
                    partial_response = from_dict(
                        data_class=API.InitLoadRemainingPairsPartialResponse,
                        data=json.loads(line),
                    )
                    load_uuid_embedding = partial_response.embeddings.uuid
                    dfs_embedding.append(_BatchedLoader.load_path(partial_response.embeddings.path, EmbeddingDataFrame))
                    load_uuid_pair = partial_response.pairs.uuid
                    dfs_pair.append(_BatchedLoader.load_path(partial_response.pairs.path, PairDataFrame))

                df_embedding = _BatchedLoader.concat(dfs_embedding, EmbeddingDataFrame)
                df_pair = _BatchedLoader.concat(dfs_pair, PairDataFrame)
                log.info("loaded batch of image pairs for test run")
                return df_embedding, df_pair
            finally:
                for uuid in [load_uuid_embedding, load_uuid_pair]:
                    _BatchedLoader.complete_load(uuid)

    def upload_pair_results(self, df_pair_result: PairResultDataFrame) -> int:
        """
        Upload image pair similarity results for a batch of pairs.

        This method should not be called until all images in the TestRun have been processed.

        All columns except for ``image_pair_id`` and ``similarity`` are optional. An empty ``similarity`` cell in a
        record indicates a pair failure (i.e. one or more of the images in the pair failed to enroll).

        For image pairs containing images with more than one embedding, a single record may be provided with the highest
        similarity score, or ``M x N`` records may be provided for each embeddings combination in the pair, when there
        are ``M`` embeddings from ``image_a`` and ``N`` embeddings from ``image_b``.

        When providing multiple records for a given image pair, use the ``embedding_a_index`` and ``embedding_b_index``
        columns to indicate which embeddings were used to compute a given similarity score. Records for a given image
        pair must be submitted in the same ``df_pair_result`` DataFrame, and **not** across multiple calls to
        ``upload_pair_results``.

        :param df_pair_result: DataFrame containing records describing the similarity score of a pair of images.
        :return: number of records successfully uploaded.
        :raises TypeValidationError: if the DataFrame failed type validation.
        :raises RemoteError: if the DataFrame was unable to be successfully ingested for any reason.
        """
        log.info("uploading pair results for test run")
        init_response = init_upload()

        df_validated = validate_df_schema(df_pair_result, PairResultDataFrameSchema)
        validate_df_record_count(df_validated)
        upload_data_frame(df_validated, BatchSize.UPLOAD_RECORDS.value, init_response.uuid)

        request = API.UploadPairResultsRequest(uuid=init_response.uuid, test_run_id=self.data.id, reset=self._reset)
        finalize_res = krequests.put(
            endpoint_path=API.Path.COMPLETE_UPLOAD_PAIR_RESULTS.value,
            data=json.dumps(dataclasses.asdict(request)),
        )
        krequests.raise_for_status(finalize_res)
        log.success("uploaded pair results for test run")
        response = from_dict(data_class=API.UploadPairResultsResponse, data=finalize_res.json())
        return response.n_uploaded


@validate_arguments(config=ValidatorConfig)
def test(model: InferenceModel, test_suite: TestSuite, reset: bool = False) -> None:
    """
    Test the provided :class:`kolena.fr.InferenceModel` on one or more provided :class:`kolena.fr.TestSuite` suites. Any
    tests already in progress for this model on these suites are resumed.

    :param model: the model being tested, implementing both ``extract`` and ``compare`` methods.
    :param test_suite: the test suite on which to test the model.
    :param reset: overwrites existing inferences if set.
    """
    test_run = TestRun(model, test_suite, reset=reset)

    try:
        log.info("starting test run")
        df_image = test_run.load_remaining_images(int(1e12))
        df_image["embedding"] = [
            model.extract(record.locator)
            for record in tqdm(df_image.itertuples(), position=0, leave=True, total=len(df_image))
        ]
        empty_columns = [  # columns not populated in this implementation
            "bounding_box",
            "landmarks_input_image",
            "landmarks",
            "quality_input_image",
            "quality",
            "acceptability",
            "fr_input_image",
            "failure_reason",
        ]
        df_image[empty_columns] = None
        df_image_result = df_image[["image_id", "embedding", *empty_columns]]
        if len(df_image_result) > 0:  # only attempt to upload if this step has not been completed
            test_run.upload_image_results(df_image_result)

        df_embedding, df_pair = test_run.load_remaining_pairs(int(1e12))
        embedding_by_id = {record.image_id: record.embedding for record in df_embedding.itertuples()}
        df_pair["similarity"] = [
            model.compare(embedding_by_id[record.image_a_id], embedding_by_id[record.image_b_id])
            for record in tqdm(df_pair.itertuples(), position=0, leave=True, total=len(df_pair))
        ]
        df_pair_result = df_pair[["image_pair_id", "similarity"]]
        if len(df_pair_result) > 0:
            test_run.upload_pair_results(df_pair_result)
        log.success("completed test run")

    except Exception as e:
        report_crash(test_run.data.id, API.Path.MARK_CRASHED.value)
        raise e
