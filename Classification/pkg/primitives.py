'''Primitives for the project

These are some derived types for the project
'''

from typing import NamedTuple, TypeVar

class PATH(NamedTuple):
    path : str

PandasDataFrame = TypeVar('pandas.core.frame.DataFrame')
