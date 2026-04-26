import json
import os
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, Optional

import attrs
import pandas as pd

from constants import (
    DATA_FILE,
    HYPERPARAMETERS_FILE,
    ID_COLUMN,
    TARGET_COLUMN,
    TEXT_COLUMN,
    TRUE_TAG_COLUMN,
    VECTORS_FILE,
)


@attrs.define
class Post:
    id: Optional[int] = None
    post: str = ""
    tags: str = ""

    def apply(self, func: Callable[[str], str]):
        self.post = func(self.post)
        self.tags = func(self.tags)


def average(numbers: Iterable[int | float], key: Optional[Callable] = lambda x: x):
    """Returns average of all numerical values in a one-dimensional Iterable or Mapping-like object"""

    if not isinstance(numbers, Iterable) or isinstance(numbers, str):
        raise TypeError(f"Expected object of type Iterable, got {type(numbers).__name__}")

    return sum((key(n) for n in numbers)) / len(numbers)


def load_posts_frame() -> pd.DataFrame:
    dataframe = pd.read_csv(DATA_FILE)

    if ID_COLUMN not in dataframe.columns:
        dataframe.insert(0, ID_COLUMN, range(len(dataframe)))

    return dataframe


def load_dataset() -> list[Post]:
    dataframe = load_posts_frame()

    return [Post(**post) for post in dataframe.to_dict(orient="records")]


def ensure_vectors_file() -> Path:
    if VECTORS_FILE.exists():
        return VECTORS_FILE

    from generate_vectors import write_vectors

    posts = load_dataset()
    write_vectors(posts, posts, VECTORS_FILE)
    return VECTORS_FILE


def load_vectors_frame() -> pd.DataFrame:
    ensure_vectors_file()
    dataframe = pd.read_csv(VECTORS_FILE)

    required_columns = {ID_COLUMN, TARGET_COLUMN}
    missing_columns = required_columns - set(dataframe.columns)
    if missing_columns:
        raise RuntimeError(
            f"Vector dataset is missing required columns: {', '.join(sorted(missing_columns))}."
        )

    return dataframe


def get_vector_feature_columns(dataframe: pd.DataFrame) -> list[str]:
    return [column_name for column_name in dataframe.columns if column_name not in {ID_COLUMN, TARGET_COLUMN}]


def load_modeling_data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    vectors = load_vectors_frame()
    posts = load_posts_frame()

    features = vectors[get_vector_feature_columns(vectors)].copy()
    target = vectors[TARGET_COLUMN].astype(str).copy()
    metadata = vectors[[ID_COLUMN, TARGET_COLUMN]].merge(
        posts[[ID_COLUMN, TEXT_COLUMN]],
        on=ID_COLUMN,
        how="left",
    )
    metadata = metadata.rename(columns={TARGET_COLUMN: TRUE_TAG_COLUMN})

    return features, target, metadata


def load_json_file(path: str | Path, default: Any) -> Any:
    path = Path(path)
    if not path.exists():
        return default

    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    except json.JSONDecodeError:
        return default


def save_json_file(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(f"{path.suffix}.tmp")

    try:
        with open(temp_path, "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=4)
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


def load_hyperparameter_results() -> dict[str, object]:
    return load_json_file(HYPERPARAMETERS_FILE, {})
