from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import cast
from typing import Generic
from typing import Optional
from typing import Type
from typing import TypeVar

import pandas as pd
import pandera as pa

from kolena._utils.dataframes.validators import validate_df_schema

# export these such that this version handling logic only needs to be applied here
try:
    # Python >= 3.8
    from typing import get_args
    from typing import get_origin
except ImportError:

    def get_args(t: Any) -> tuple:
        return getattr(t, "__args__", ())

    def get_origin(t: Any) -> Optional[Type]:
        return getattr(t, "__origin__", None)


TDataFrame = TypeVar("TDataFrame", bound="LoadableDataFrame")
TSchema = TypeVar("TSchema", bound=pa.SchemaModel)


class LoadableDataFrame(ABC, Generic[TDataFrame]):
    @classmethod
    @abstractmethod
    def get_schema(cls) -> Type[TSchema]:
        raise NotImplementedError

    @classmethod
    def construct_empty(cls) -> TDataFrame:
        df = pd.DataFrame({key: [] for key in cls.get_schema().to_schema().columns.keys()})
        return cast(TDataFrame, validate_df_schema(df, cls.get_schema(), trusted=True))

    @classmethod
    def from_serializable(cls, df: pd.DataFrame) -> TDataFrame:
        return cast(TDataFrame, validate_df_schema(df, cls.get_schema(), trusted=True))
