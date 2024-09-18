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
import os
import re
import tarfile
import xml.etree.ElementTree as et
from argparse import ArgumentParser
from argparse import Namespace
from collections.abc import Iterator
from typing import Any
from typing import Dict

import pandas as pd
from named_entity_recognition.constants import DATASET
from named_entity_recognition.tag_map import TagMap
from tqdm import tqdm

from kolena.annotation import LabeledTextSegment
from kolena.dataset import upload_dataset


def read_tar_gz(fpath: str) -> Iterator[Dict[str, Any]]:
    """
    Read .tar.gz file
    """
    tf = tarfile.open(fpath, "r:gz")

    for tf_member in tf.getmembers():
        file_object = tf.extractfile(tf_member)
        name = tf_member.name
        file_name = os.path.basename(name).split(".")[0]
        if re.search(r"\.xml", name) is not None:
            xml_flag = True
        else:
            xml_flag = False
        yield {
            "file_object": file_object,
            "file_name": file_name,
            "xml_flag": xml_flag,
        }


def read_dataset_xml_file(file_object: Any, file_name: str) -> Dict[str, Any]:
    xmldoc = et.parse(file_object).getroot()
    entities = xmldoc.findall("TAGS")[0]
    text = xmldoc.findall("TEXT")[0].text
    phi = []
    for entity in entities:
        phi.append(
            LabeledTextSegment(
                text_field="text",
                label=entity.attrib["TYPE"],
                start=int(entity.attrib["start"]),
                end=int(entity.attrib["end"]),
                id=entity.attrib["id"],  # type: ignore[call-arg]
                comment=entity.attrib["comment"],  # type: ignore[call-arg]
            ),
        )

    document = {"document_id": file_name, "text": text, "phi": phi}
    return document


def run(args: Namespace) -> None:
    documents = []
    for x in tqdm(read_tar_gz(args.dataset_targz_file)):
        xml_flag = x["xml_flag"]
        if xml_flag:
            documents.append(read_dataset_xml_file(file_object=x["file_object"], file_name=x["file_name"]))

    df_dataset = pd.DataFrame(documents)

    # update the tags
    tag_map = TagMap().get_proposed_tags()
    phis = []
    for record in df_dataset.itertuples():
        groundtruth = []
        for phi in record.phi:
            details = phi._to_dict()
            try:
                details["label"] = tag_map[phi.label]
            except KeyError:
                continue
            groundtruth.append(LabeledTextSegment(**details))

        phis.append(groundtruth)

    df_dataset["phi"] = phis
    upload_dataset(args.dataset, df_dataset)


def main() -> None:
    ap = ArgumentParser()
    ap.add_argument(
        "--dataset-targz-file",
        type=str,
        help="Specify the location of the dataset tar.gz file",
        default="./data/testing-PHI-Gold-fixed.tar.gz",
    )
    ap.add_argument(
        "--dataset",
        type=str,
        default=DATASET,
        help="Optionally specify a custom dataset name to upload.",
    )
    run(ap.parse_args())


if __name__ == "__main__":
    main()
