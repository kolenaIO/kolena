from abc import ABC
from abc import abstractmethod

class BaseDataFormat(ABC):
    def __init__(self) -> None:
        pass
    
    @abstractmethod
    def load_data(self, file_path: str) -> None:
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
