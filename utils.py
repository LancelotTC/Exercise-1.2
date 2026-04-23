import attrs
import pandas as pd
from constants import *
from typing import Optional
from functools import partial
from collections.abc import Callable, Iterable


@attrs.define
class Post:
    post: str = ""
    tags: str = ""

    def apply(self, func: Callable[[str], str]):
        self.post = func(self.post)
        self.tags = func(self.tags)


def average(numbers: Iterable[int | float], key: Optional[Callable] = lambda x: x) -> int | float:
    """Returns average of all numerical values in a one-dimensional Iterable or Mapping-like object"""

    if not isinstance(numbers, Iterable) or isinstance(numbers, str):
        raise TypeError(f"Expected object of type Iterable, got {type(numbers).__name__}")

    return sum((key(n) for n in numbers)) / len(numbers)


def load_dataset():
    df = pd.read_csv(DATA_FILE)

    return [Post(**post) for post in df.to_dict(orient="records")]
