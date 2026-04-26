from pathlib import Path

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from constants import (
    HYPERPARAMETERS_FILE,
    IS_CORRECT_COLUMN,
    PREDICTIONS_DIR,
    PREDICTED_PROBABILITY_COLUMN,
    PREDICTED_TAG_COLUMN,
    RANDOM_SEED,
    TRUE_TAG_COLUMN,
    VALIDATION_SIZE,
)
from utils import load_hyperparameter_results, load_modeling_data


def build_model(model_name: str, best_params: dict):
    if model_name == "LogisticRegression":
        return LogisticRegression(max_iter=4000, random_state=RANDOM_SEED, **best_params)
    if model_name == "RandomForestClassifier":
        return RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=-1, **best_params)
    if model_name == "DecisionTreeClassifier":
        return DecisionTreeClassifier(random_state=RANDOM_SEED, **best_params)
    if model_name == "HistGradientBoostingClassifier":
        return HistGradientBoostingClassifier(random_state=RANDOM_SEED, **best_params)
    if model_name == "XGBClassifier":
        return XGBClassifier(
            random_state=RANDOM_SEED,
            n_jobs=-1,
            objective="multi:softprob",
            eval_metric="mlogloss",
            tree_method="hist",
            **best_params,
        )

    raise RuntimeError(f"Unsupported model '{model_name}' in {HYPERPARAMETERS_FILE}.")


def prediction_output_path(model_name: str, filename: str) -> Path:
    output_dir = PREDICTIONS_DIR / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / filename


def build_prediction_frame(metadata, predicted_tags, predicted_probabilities):
    output = metadata.copy()
    output[PREDICTED_TAG_COLUMN] = predicted_tags
    output[PREDICTED_PROBABILITY_COLUMN] = predicted_probabilities
    output[IS_CORRECT_COLUMN] = output[TRUE_TAG_COLUMN] == output[PREDICTED_TAG_COLUMN]
    return output


def write_prediction_file(path: Path, dataframe) -> Path:
    dataframe.to_csv(path, index=False)
    return path


def predict_with_confidence(model, features):
    probabilities = model.predict_proba(features)
    predicted_tags = model.classes_[probabilities.argmax(axis=1)]
    predicted_probabilities = probabilities.max(axis=1)
    return predicted_tags, predicted_probabilities


def fit_and_predict(model_name: str, model, X_train, y_train, X_predict):
    if model_name != "XGBClassifier":
        model.fit(X_train, y_train)
        return predict_with_confidence(model, X_predict)

    encoder = LabelEncoder()
    encoded_target = encoder.fit_transform(y_train)
    model.fit(X_train, encoded_target)
    predicted_probabilities = model.predict_proba(X_predict).max(axis=1)
    predicted_tags = encoder.inverse_transform(model.predict(X_predict).astype(int))
    return predicted_tags, predicted_probabilities


def run_individual_predictions():
    results = load_hyperparameter_results()
    if not results:
        raise RuntimeError(f"No hyperparameter results found in '{HYPERPARAMETERS_FILE}'.")

    features, target, metadata = load_modeling_data()
    X_train, X_val, y_train, y_val, _, metadata_val = train_test_split(
        features,
        target,
        metadata,
        test_size=VALIDATION_SIZE,
        stratify=target,
        random_state=RANDOM_SEED,
    )

    for model_name, result in results.items():
        model = build_model(model_name, result["best_params"])
        validation_tags, validation_probabilities = fit_and_predict(model_name, model, X_train, y_train, X_val)
        validation_output = build_prediction_frame(metadata_val, validation_tags, validation_probabilities)
        validation_output_path = write_prediction_file(
            prediction_output_path(model_name, f"{model_name}_validation_preds.csv"),
            validation_output,
        )

        validation_accuracy = accuracy_score(y_val, validation_tags)
        validation_macro_f1 = f1_score(y_val, validation_tags, average="macro")

        full_model = build_model(model_name, result["best_params"])
        all_tags, all_probabilities = fit_and_predict(model_name, full_model, features, target, features)
        all_output = build_prediction_frame(metadata, all_tags, all_probabilities)
        all_output_path = write_prediction_file(
            prediction_output_path(model_name, f"{model_name}_all_rows_preds.csv"),
            all_output,
        )

        print(f"{model_name} validation accuracy: {validation_accuracy:.4f}")
        print(f"{model_name} validation macro F1: {validation_macro_f1:.4f}")
        print(f"Validation predictions written to {validation_output_path}")
        print(f"All-row predictions written to {all_output_path}")


if __name__ == "__main__":
    run_individual_predictions()
