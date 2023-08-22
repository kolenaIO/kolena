from typing import Optional
from kolena._experimental.data_ingestion._data_format.base import BaseDataFormat
from kolena._experimental.data_ingestion._types import DataIngestionConfig

def _ingest_data(data_format_instance: BaseDataFormat) -> None:
    data_format_instance.load_data()
    data_format_instance.process_data()
    data_format_instance.save_data()

def get_target_data_format_class() -> BaseDataFormat:
    """guess the target data format class"""
    return BaseDataFormat 

def ingest_data(config: DataIngestionConfig, target_data_format_class: Optional[BaseDataFormat] = None) -> None:
    if target_data_format_class is None:
        target_data_format_class = get_target_data_format_class()
    target_data_format_instance = target_data_format_class(config)
    return _ingest_data(target_data_format_instance)
