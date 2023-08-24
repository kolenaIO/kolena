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
from abc import ABC
from abc import abstractmethod

from kolena._experimental.data_ingestion._types import DataIngestionConfig


class BaseDataFormat(ABC):
    def __init__(self, config: DataIngestionConfig) -> None:
        self.config = config

    @abstractmethod
    def load_data(self) -> None:
        """Load data from a file"""
        raise NotImplementedError

    @abstractmethod
    def process_data(self) -> None:
        """Process the loaded data"""
        raise NotImplementedError

    @abstractmethod
    def save_data(self) -> None:
        """Save data to Kolena system"""
        raise NotImplementedError

    @abstractmethod
    def get_workflow(self) -> None:
        """Get workflow related classes"""
        raise NotImplementedError

    def ingest_data(self) -> None:
        """Entry point to ingest data"""
        self.load_data()
        self.process_data()
        self.save_data()
