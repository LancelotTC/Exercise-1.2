from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
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
from utils import hyperparameter_result_matches, load_hyperparameter_results, load_modeling_data


STACKING_FOLDS = 3
SUPPORTED_MODEL_NAMES = (
    "LogisticRegression",
    "DecisionTreeClassifier",
    "XGBClassifier",
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


def ensemble_output_path(model_name: str, filename: str) -> Path:
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
        "recall": recall_score(true_tags, predicted_tags, labels=labels, average="macro", zero_division=0),
        "specificity": macro_specificity_score(matrix),
        "f1": f1_score(true_tags, predicted_tags, labels=labels, average="macro", zero_division=0),
    }


def write_confusion_matrix_files(model_name: str, labels: list[str], metrics: dict):
    matrix = metrics["matrix"]
    matrix_frame = pd.DataFrame(matrix, index=labels, columns=labels)
    matrix_frame.index.name = TRUE_TAG_COLUMN
    matrix_frame.columns.name = PREDICTED_TAG_COLUMN

    matrix_csv_path = ensemble_output_path(model_name, f"{model_name}_validation_confusion_matrix.csv")
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
        f"Specificity (macro): {metrics['specificity']:.4f} | F1 (macro): {metrics['f1']:.4f}"
    )
    figure.tight_layout(rect=(0, 0.07, 1, 1))
    figure.text(
        0.5,
        0.015,
        metrics_text,
        ha="center",
        va="bottom",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#d1d5db"},
    )

    matrix_plot_path = ensemble_output_path(model_name, f"{model_name}_validation_confusion_matrix.png")
    figure.savefig(matrix_plot_path, dpi=150, bbox_inches="tight")
    plt.close(figure)

    return matrix_csv_path, matrix_plot_path


def estimator_name(model_name: str) -> str:
    return "".join(character.lower() for character in model_name if character.isalnum())


def build_base_estimators(results: dict):
    if len(results) < 2:
        raise RuntimeError("Need at least two tuned models in hyperparameters.json for stacking ensembles.")

    return [
        (estimator_name(model_name), build_model(model_name, results[model_name]["best_params"]))
        for model_name in results
    ]


def build_ensemble_models(results: dict):
    cv = StratifiedKFold(n_splits=STACKING_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    base_estimators = build_base_estimators(results)

    return {
        "StackingClassifier": StackingClassifier(
            estimators=base_estimators,
            final_estimator=LogisticRegression(max_iter=4000, random_state=RANDOM_SEED, class_weight="balanced"),
            stack_method="predict_proba",
            cv=cv,
            n_jobs=-1,
        ),
        "StackingClassifierPassthrough": StackingClassifier(
            estimators=base_estimators,
            final_estimator=LogisticRegression(max_iter=4000, random_state=RANDOM_SEED, class_weight="balanced"),
            stack_method="predict_proba",
            passthrough=True,
            cv=cv,
            n_jobs=-1,
        ),
        "SoftVotingClassifier": VotingClassifier(
            estimators=base_estimators,
            voting="soft",
            n_jobs=-1,
        ),
        "HardVotingClassifier": VotingClassifier(
            estimators=base_estimators,
            voting="hard",
            n_jobs=-1,
        ),
    }


def predict_with_confidence(model, features):
    predicted_labels = model.predict(features)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)
        return predicted_labels, probabilities.max(axis=1)

    base_predictions = np.column_stack([estimator.predict(features) for estimator in model.estimators_])
    vote_confidence = (base_predictions == predicted_labels[:, None]).mean(axis=1)
    return predicted_labels, vote_confidence


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


def run_stacking_predictions():
    features, target, metadata = load_modeling_data()
    results = load_compatible_results(features.columns)

    if len(results) < 2:
        raise RuntimeError(
            f"Need at least two compatible tuned models in '{HYPERPARAMETERS_FILE}' for stacking. "
            "Run step 2 with more than one active model first."
        )

    label_encoder = LabelEncoder()
    encoded_target = label_encoder.fit_transform(target)
    labels = list(label_encoder.classes_)

    train_features, validation_features, train_target, validation_target, _, validation_metadata = train_test_split(
        features,
        encoded_target,
        metadata,
        test_size=VALIDATION_SIZE,
        stratify=encoded_target,
        random_state=RANDOM_SEED,
    )

    print("Base models used:")
    for model_name in results:
        print(f"- {model_name}")

    validation_models = build_ensemble_models(results)
    full_models = build_ensemble_models(results)

    for ensemble_name, validation_model in validation_models.items():
        validation_codes, validation_probabilities = predict_with_confidence(
            validation_model.fit(train_features, train_target),
            validation_features,
        )
        validation_tags = label_encoder.inverse_transform(validation_codes.astype(int))
        true_validation_tags = label_encoder.inverse_transform(validation_target)
        validation_output = build_prediction_frame(validation_metadata, validation_tags, validation_probabilities)
        validation_output_path = ensemble_output_path(ensemble_name, f"{ensemble_name}_validation_preds.csv")
        validation_output.to_csv(validation_output_path, index=False)

        validation_metrics = calculate_validation_metrics(labels, true_validation_tags, validation_tags)
        matrix_csv_path, matrix_plot_path = write_confusion_matrix_files(ensemble_name, labels, validation_metrics)

        full_model = full_models[ensemble_name]
        all_codes, all_probabilities = predict_with_confidence(full_model.fit(features, encoded_target), features)
        all_tags = label_encoder.inverse_transform(all_codes.astype(int))
        all_output = build_prediction_frame(metadata, all_tags, all_probabilities)
        all_output_path = ensemble_output_path(ensemble_name, f"{ensemble_name}_all_rows_preds.csv")
        all_output.to_csv(all_output_path, index=False)

        print(f"\n{ensemble_name} validation accuracy: {validation_metrics['accuracy']:.4f}")
        print(f"{ensemble_name} validation macro precision: {validation_metrics['precision']:.4f}")
        print(f"{ensemble_name} validation macro recall: {validation_metrics['recall']:.4f}")
        print(f"{ensemble_name} validation macro specificity: {validation_metrics['specificity']:.4f}")
        print(f"{ensemble_name} validation macro F1: {validation_metrics['f1']:.4f}")
        print(f"Validation predictions written to {validation_output_path}")
        print(f"Validation confusion matrix written to {matrix_csv_path}")
        print(f"Validation confusion matrix plot written to {matrix_plot_path}")
        print(f"All-row predictions written to {all_output_path}")


if __name__ == "__main__":
    run_stacking_predictions()
