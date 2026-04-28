from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
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
from utils import (
    get_feature_signature,
    hyperparameter_result_matches,
    load_hyperparameter_results,
    load_modeling_data,
)


SUPPORTED_MODEL_NAMES = (
    "LogisticRegression",
    "DecisionTreeClassifier",
    # "XGBClassifier",
)


def build_model(model_name: str, best_params: dict):
    match model_name:
        case "LogisticRegression":
            return LogisticRegression(max_iter=4000, random_state=RANDOM_SEED, **best_params)
        case "DecisionTreeClassifier":
            return DecisionTreeClassifier(random_state=RANDOM_SEED, **best_params)
        case "XGBClassifier":
            return XGBClassifier(
                random_state=RANDOM_SEED,
                n_jobs=-1,
                objective="multi:softprob",
                eval_metric="mlogloss",
                tree_method="hist",
                **best_params,
            )
        case _:
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


def macro_specificity_score(matrix) -> float:
    total = matrix.sum()
    scores = []

    for index in range(len(matrix)):
        true_positive = matrix[index, index]
        false_negative = matrix[index, :].sum() - true_positive
        false_positive = matrix[:, index].sum() - true_positive
        true_negative = total - true_positive - false_negative - false_positive
        denominator = true_negative + false_positive
        scores.append(0.0 if denominator == 0 else true_negative / denominator)

    return sum(scores) / len(scores)


def calculate_validation_metrics(labels: list[str], true_tags, predicted_tags):
    matrix = confusion_matrix(true_tags, predicted_tags, labels=labels)
    return {
        "matrix": matrix,
        "accuracy": accuracy_score(true_tags, predicted_tags),
        "precision": precision_score(true_tags, predicted_tags, labels=labels, average="macro", zero_division=0),
        "precision_micro": precision_score(true_tags, predicted_tags, labels=labels, average="micro", zero_division=0),
        "recall": recall_score(true_tags, predicted_tags, labels=labels, average="macro", zero_division=0),
        "recall_micro": recall_score(true_tags, predicted_tags, labels=labels, average="micro", zero_division=0),
        "specificity": macro_specificity_score(matrix),
        "f1": f1_score(true_tags, predicted_tags, labels=labels, average="macro", zero_division=0),
        "f1_micro": f1_score(true_tags, predicted_tags, labels=labels, average="micro", zero_division=0),
    }


def write_confusion_matrix_files(model_name: str, labels: list[str], metrics: dict):
    matrix = metrics["matrix"]
    matrix_frame = pd.DataFrame(matrix, index=labels, columns=labels)
    matrix_frame.index.name = TRUE_TAG_COLUMN
    matrix_frame.columns.name = PREDICTED_TAG_COLUMN

    matrix_csv_path = prediction_output_path(model_name, f"{model_name}_validation_confusion_matrix.csv")
    matrix_frame.to_csv(matrix_csv_path)

    figure_size = max(8, len(labels) * 0.6)
    figure, axis = plt.subplots(figsize=(figure_size, figure_size))
    ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=labels).plot(
        ax=axis,
        cmap="Blues",
        colorbar=True,
        xticks_rotation=45,
        include_values=False,
    )
    axis.set_title(f"{model_name} Validation Confusion Matrix")
    metrics_text = (
        f"Accuracy: {metrics['accuracy']:.4f} | Precision (macro): {metrics['precision']:.4f} | "
        f"Recall (macro): {metrics['recall']:.4f}\n"
        f"Specificity (macro): {metrics['specificity']:.4f} | F1 (macro): {metrics['f1']:.4f}\n"
        f"Micro P/R/F1: {metrics['precision_micro']:.4f} / {metrics['recall_micro']:.4f} / {metrics['f1_micro']:.4f} | "
        f"Train time: {metrics['train_seconds']:.2f}s"
    )
    figure.tight_layout(rect=(0, 0.11, 1, 1))
    figure.text(
        0.5,
        0.015,
        metrics_text,
        ha="center",
        va="bottom",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#d1d5db"},
    )

    matrix_plot_path = prediction_output_path(model_name, f"{model_name}_validation_confusion_matrix.png")
    figure.savefig(matrix_plot_path, dpi=150, bbox_inches="tight")
    plt.close(figure)

    return matrix_csv_path, matrix_plot_path


def predict_with_confidence(model, features):
    probabilities = model.predict_proba(features)
    predicted_tags = model.classes_[probabilities.argmax(axis=1)]
    predicted_probabilities = probabilities.max(axis=1)
    return predicted_tags, predicted_probabilities


def fit_and_predict(model_name: str, model, train_features, train_target, predict_features):
    started_at = perf_counter()

    if model_name != "XGBClassifier":
        model.fit(train_features, train_target)
        predicted_tags, predicted_probabilities = predict_with_confidence(model, predict_features)
        return predicted_tags, predicted_probabilities, perf_counter() - started_at

    encoder = LabelEncoder()
    encoded_target = encoder.fit_transform(train_target)
    model.fit(train_features, encoded_target)
    predicted_probabilities = model.predict_proba(predict_features).max(axis=1)
    predicted_tags = encoder.inverse_transform(model.predict(predict_features).astype(int))
    return predicted_tags, predicted_probabilities, perf_counter() - started_at


def load_compatible_results(feature_columns):
    loaded_results = load_hyperparameter_results()
    compatible_results = {}

    for model_name in SUPPORTED_MODEL_NAMES:
        if model_name not in loaded_results:
            continue

        result = loaded_results[model_name]
        if hyperparameter_result_matches(result, feature_columns):
            compatible_results[model_name] = result

    return compatible_results


def build_incompatibility_message(feature_columns) -> str:
    loaded_results = load_hyperparameter_results()
    if not loaded_results:
        return f"No saved results found in '{HYPERPARAMETERS_FILE}'. Run step 2 first."

    current_feature_count = len(feature_columns)
    current_feature_signature = get_feature_signature(feature_columns)
    details = []

    for model_name in SUPPORTED_MODEL_NAMES:
        if model_name not in loaded_results:
            continue

        result = loaded_results[model_name]
        reasons = []

        saved_feature_count = result.get("feature_columns")
        if saved_feature_count != current_feature_count:
            reasons.append(f"feature_columns={saved_feature_count}, current={current_feature_count}")

        saved_feature_signature = result.get("feature_signature")
        if saved_feature_signature != current_feature_signature:
            if saved_feature_signature is None:
                reasons.append("missing feature_signature")
            else:
                reasons.append("feature_signature mismatch")

        if not reasons:
            reasons.append("unsupported or unknown mismatch")

        details.append(f"{model_name}: " + "; ".join(reasons))

    if not details:
        return f"No compatible hyperparameter results found in '{HYPERPARAMETERS_FILE}'. " "Run step 2 first."

    return (
        f"No compatible hyperparameter results found in '{HYPERPARAMETERS_FILE}'.\n"
        f"Current feature columns: {current_feature_count}\n"
        f"Incompatibilities:\n- " + "\n- ".join(details) + "\n"
        "Run step 2 again so the saved results match the current vector file."
    )


def run_individual_predictions():
    features, target, metadata = load_modeling_data()
    results = load_compatible_results(features.columns)

    if not results:
        raise RuntimeError(build_incompatibility_message(features.columns))

    labels = sorted(target.unique())
    train_features, validation_features, train_target, validation_target, _, validation_metadata = train_test_split(
        features,
        target,
        metadata,
        test_size=VALIDATION_SIZE,
        stratify=target,
        random_state=RANDOM_SEED,
    )

    for model_name, result in results.items():
        model = build_model(model_name, result["best_params"])
        validation_tags, validation_probabilities, train_seconds = fit_and_predict(
            model_name,
            model,
            train_features,
            train_target,
            validation_features,
        )

        validation_output = build_prediction_frame(validation_metadata, validation_tags, validation_probabilities)
        validation_output_path = prediction_output_path(model_name, f"{model_name}_validation_preds.csv")
        validation_output.to_csv(validation_output_path, index=False)

        validation_metrics = calculate_validation_metrics(labels, validation_target, validation_tags)
        validation_metrics["train_seconds"] = train_seconds
        matrix_csv_path, matrix_plot_path = write_confusion_matrix_files(model_name, labels, validation_metrics)

        full_model = build_model(model_name, result["best_params"])
        all_tags, all_probabilities, _ = fit_and_predict(model_name, full_model, features, target, features)
        all_output = build_prediction_frame(metadata, all_tags, all_probabilities)
        all_output_path = prediction_output_path(model_name, f"{model_name}_all_rows_preds.csv")
        all_output.to_csv(all_output_path, index=False)

        print(f"{model_name} validation accuracy: {validation_metrics['accuracy']:.4f}")
        print(f"{model_name} validation macro precision: {validation_metrics['precision']:.4f}")
        print(f"{model_name} validation micro precision: {validation_metrics['precision_micro']:.4f}")
        print(f"{model_name} validation macro recall: {validation_metrics['recall']:.4f}")
        print(f"{model_name} validation micro recall: {validation_metrics['recall_micro']:.4f}")
        print(f"{model_name} validation macro specificity: {validation_metrics['specificity']:.4f}")
        print(f"{model_name} validation macro F1: {validation_metrics['f1']:.4f}")
        print(f"{model_name} validation micro F1: {validation_metrics['f1_micro']:.4f}")
        print(f"{model_name} train time: {validation_metrics['train_seconds']:.2f}s")
        print(f"Validation predictions written to {validation_output_path}")
        print(f"Validation confusion matrix written to {matrix_csv_path}")
        print(f"Validation confusion matrix plot written to {matrix_plot_path}")
        print(f"All-row predictions written to {all_output_path}")


if __name__ == "__main__":
    run_individual_predictions()
