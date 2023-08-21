from kolena._experimental.data_ingestion._data_format.base import BaseDataFormat

class CocoDataFormat(BaseDataFormat):
    def __init__(self) -> None:
        super().__init__()
    
    def load_data(self, file_path: str) -> None:
        ...

    def process_data(self) -> None:
        ...

    def save_data(self) -> None:
        ...
 