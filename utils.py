import importlib.util
import hashlib
import json
import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

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


@dataclass(slots=True)
class Post:
    id: Optional[int] = None
    post: str = ""
    tags: str = ""

    def apply(self, func: Callable[[str], str]):
        self.post = func(self.post)
        self.tags = func(self.tags)


def load_posts_frame() -> pd.DataFrame:
    dataframe = pd.read_csv(DATA_FILE)

    if ID_COLUMN not in dataframe.columns:
        dataframe.insert(0, ID_COLUMN, range(len(dataframe)))

    return dataframe


def load_dataset() -> list[Post]:
    dataframe = load_posts_frame()

    return [Post(**post) for post in dataframe.to_dict(orient="records")]


@lru_cache(maxsize=1)
def load_vector_generation_module():
    module_path = Path(__file__).with_name("1_generate_vectors.py")
    spec = importlib.util.spec_from_file_location("pipeline_generate_vectors", module_path)

    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load vector generation module from '{module_path}'.")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def get_feature_signature(column_names: Iterable[str]) -> str:
    joined_names = "\n".join(column_names)
    return hashlib.sha256(joined_names.encode("utf-8")).hexdigest()


def hyperparameter_result_matches(
    result: dict[str, Any],
    feature_columns: list[str] | pd.Index,
    training_rows: int | None = None,
) -> bool:
    if result.get("feature_columns") != len(feature_columns):
        return False

    if result.get("feature_signature") != get_feature_signature(feature_columns):
        return False

    if training_rows is not None and result.get("training_rows") != training_rows:
        return False

    return True


def ensure_vectors_file() -> Path:
    if VECTORS_FILE.exists():
        return VECTORS_FILE

    vector_module = load_vector_generation_module()
    posts = load_dataset()
    vector_module.write_vectors(posts, posts, VECTORS_FILE)
    return VECTORS_FILE


def load_vectors_frame() -> pd.DataFrame:
    ensure_vectors_file()
    dataframe = pd.read_csv(VECTORS_FILE)

    required_columns = {ID_COLUMN, TARGET_COLUMN}
    missing_columns = required_columns - set(dataframe.columns)
    if missing_columns:
        raise RuntimeError(f"Vector dataset is missing required columns: {', '.join(sorted(missing_columns))}.")

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
