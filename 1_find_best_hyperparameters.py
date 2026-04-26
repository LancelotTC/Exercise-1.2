from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from skopt import BayesSearchCV
from skopt.space import Categorical, Integer, Real
from xgboost import XGBClassifier

from constants import (
    HYPERPARAMETER_SEARCH_FOLDS,
    HYPERPARAMETER_SEARCH_ITERATIONS,
    HYPERPARAMETER_SEARCH_JOBS,
    HYPERPARAMETER_SEARCH_ROWS,
    HYPERPARAMETERS_FILE,
    RANDOM_SEED,
)
from utils import load_hyperparameter_results, load_modeling_data, save_json_file


def sample_training_rows(features, target):
    if HYPERPARAMETER_SEARCH_ROWS is None or len(features) <= HYPERPARAMETER_SEARCH_ROWS:
        return features, target

    if HYPERPARAMETER_SEARCH_ROWS < target.nunique():
        raise RuntimeError("HYPERPARAMETER_SEARCH_ROWS must be at least the number of classes.")

    sampled_features, _, sampled_target, _ = train_test_split(
        features,
        target,
        train_size=HYPERPARAMETER_SEARCH_ROWS,
        stratify=target,
        random_state=RANDOM_SEED,
    )
    return sampled_features, sampled_target


def build_searches():
    return {
        # "LogisticRegression": (
        #     LogisticRegression(max_iter=4000, random_state=RANDOM_SEED),
        #     {
        #         "C": Real(1e-4, 1e3, prior="log-uniform"),
        #         "class_weight": Categorical([None, "balanced"]),
        #         "fit_intercept": Categorical([True, False]),
        #         "tol": Real(1e-6, 1e-2, prior="log-uniform"),
        #     },
        # ),
        # "RandomForestClassifier": (
        #     RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=HYPERPARAMETER_SEARCH_JOBS),
        #     {
        #         "n_estimators": Integer(100, 2500),
        #         "max_depth": Categorical([None, 4, 8, 12, 20, 40, 80]),
        #         "min_samples_split": Integer(2, 200),
        #         "min_samples_leaf": Integer(1, 100),
        #         "max_features": Real(0.05, 1.0),
        #         "bootstrap": Categorical([True, False]),
        #         "criterion": Categorical(["gini", "entropy", "log_loss"]),
        #         "class_weight": Categorical([None, "balanced"]),
        #     },
        # ),
        "DecisionTreeClassifier": (
            DecisionTreeClassifier(random_state=RANDOM_SEED),
            {
                "criterion": Categorical(["gini", "entropy", "log_loss"]),
                "splitter": Categorical(["best", "random"]),
                "max_depth": Categorical([None, 2, 4, 6, 8, 12, 20, 40, 80]),
                "min_samples_split": Integer(2, 500),
                "min_samples_leaf": Integer(1, 250),
                "max_features": Categorical([None, "sqrt", "log2"]),
                "class_weight": Categorical([None, "balanced"]),
                "min_impurity_decrease": Real(0.0, 0.05),
                "ccp_alpha": Real(1e-10, 1.0, prior="log-uniform"),
            },
        ),
        # "HistGradientBoostingClassifier": (
        #     HistGradientBoostingClassifier(random_state=RANDOM_SEED),
        #     {
        #         "learning_rate": Real(1e-3, 0.5, prior="log-uniform"),
        #         "max_iter": Integer(100, 3000),
        #         "max_depth": Categorical([None, 2, 3, 5, 8, 12, 20]),
        #         "max_leaf_nodes": Integer(7, 255),
        #         "min_samples_leaf": Integer(2, 500),
        #         "l2_regularization": Real(1e-10, 100.0, prior="log-uniform"),
        #         "max_features": Real(0.2, 1.0),
        #         "class_weight": Categorical([None, "balanced"]),
        #     },
        # ),
        # "XGBClassifier": (
        #     XGBClassifier(
        #         random_state=RANDOM_SEED,
        #         n_jobs=HYPERPARAMETER_SEARCH_JOBS,
        #         objective="multi:softprob",
        #         eval_metric="mlogloss",
        #         tree_method="hist",
        #     ),
        #     {
        #         "n_estimators": Integer(100, 3000),
        #         "learning_rate": Real(1e-3, 0.5, prior="log-uniform"),
        #         "max_depth": Integer(2, 20),
        #         "min_child_weight": Integer(1, 50),
        #         "subsample": Real(0.3, 1.0),
        #         "colsample_bytree": Real(0.3, 1.0),
        #         "gamma": Real(1e-10, 10.0, prior="log-uniform"),
        #         "reg_alpha": Real(1e-10, 100.0, prior="log-uniform"),
        #         "reg_lambda": Real(1e-3, 100.0, prior="log-uniform"),
        #         "grow_policy": Categorical(["depthwise", "lossguide"]),
        #     },
        # ),
    }


def run_hyperparameter_search():
    features, target, _ = load_modeling_data()
    features, target = sample_training_rows(features, target)

    cv = StratifiedKFold(
        n_splits=HYPERPARAMETER_SEARCH_FOLDS,
        shuffle=True,
        random_state=RANDOM_SEED,
    )
    results = load_hyperparameter_results()

    print(f"Training rows: {len(features)}")
    print(f"Feature columns: {len(features.columns)}")

    for model_name, (model, search_space) in build_searches().items():
        search_target = target
        if model_name == "XGBClassifier":
            search_target = LabelEncoder().fit_transform(target)
        print(f"\nSearching {model_name} ({HYPERPARAMETER_SEARCH_ITERATIONS} iterations)")

        search = BayesSearchCV(
            estimator=model,
            search_spaces=search_space,
            n_iter=HYPERPARAMETER_SEARCH_ITERATIONS,
            scoring="f1_macro",
            cv=cv,
            random_state=RANDOM_SEED,
            n_jobs=HYPERPARAMETER_SEARCH_JOBS,
            refit=True,
            verbose=0,
        )
        search.fit(features, search_target)

        best_score = float(search.best_score_)
        previous_result = results.get(model_name, {})
        previous_rows = previous_result.get("training_rows")
        previous_score = (
            float(previous_result.get("best_score", float("-inf"))) if previous_rows == len(features) else float("-inf")
        )

        if best_score <= previous_score:
            print(f"Best macro F1: {best_score:.4f} (kept existing result)")
            continue

        results[model_name] = {
            "best_score": best_score,
            "best_params": search.best_params_,
            "training_rows": len(features),
            "feature_columns": len(features.columns),
            "scoring": "f1_macro",
        }
        save_json_file(HYPERPARAMETERS_FILE, results)
        print(f"Best macro F1: {best_score:.4f}")

    return results


if __name__ == "__main__":
    run_hyperparameter_search()
