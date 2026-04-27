from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from skopt import BayesSearchCV
from skopt.callbacks import DeltaYStopper
from skopt.space import Categorical, Integer, Real
from xgboost import XGBClassifier

from constants import (
    HYPERPARAMETER_SEARCH_DELTA_Y,
    HYPERPARAMETER_SEARCH_DELTA_Y_N_BEST,
    HYPERPARAMETER_SEARCH_FOLDS,
    HYPERPARAMETER_SEARCH_ITERATIONS,
    HYPERPARAMETER_SEARCH_JOBS,
    HYPERPARAMETER_SEARCH_ROWS,
    HYPERPARAMETERS_FILE,
    RANDOM_SEED,
)
from utils import load_hyperparameter_results, load_modeling_data, save_json_file
from tqdm.auto import tqdm
from skopt.callbacks import DeltaYStopper


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
        "LogisticRegression": (
            LogisticRegression(max_iter=4000, random_state=RANDOM_SEED),
            {
                "C": Real(1e-4, 1e3, prior="log-uniform"),
                "class_weight": Categorical([None, "balanced"]),
                "fit_intercept": Categorical([True, False]),
                # "solver": Categorical(["lbfgs", "newton-cg", "newton-cholesky", "sag", "saga"]),
                "tol": Real(1e-6, 1e-2, prior="log-uniform"),
            },
        ),
        # "DecisionTreeClassifier": (
        #     DecisionTreeClassifier(random_state=RANDOM_SEED),
        #     {
        #         "criterion": Categorical(["gini", "entropy", "log_loss"]),
        #         "splitter": Categorical(["best", "random"]),
        #         "max_depth": Categorical([None, 6, 8, 12, 20, 40, 80, 120]),
        #         "max_leaf_nodes": Categorical([None, 16, 32, 64, 128, 256, 512, 1024]),
        #         "min_samples_split": Categorical([2, 3, 4, 5, 8, 12, 20, 40, 80, 160, 320]),
        #         "min_samples_leaf": Categorical([1, 2, 3, 4, 5, 8, 12, 20, 40, 80]),
        #         "min_weight_fraction_leaf": Categorical([0.0, 1e-5, 1e-4, 1e-3, 5e-3, 1e-2]),
        #         "max_features": Categorical([None, "sqrt", "log2", 0.25, 0.5, 0.75]),
        #         "min_impurity_decrease": Categorical([0.0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]),
        #         "ccp_alpha": Categorical([0.0, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2]),
        #     },
        # ),
        # "KNeighborsClassifier": (
        #     KNeighborsClassifier(n_jobs=HYPERPARAMETER_SEARCH_JOBS),
        #     {
        #         "n_neighbors": Integer(1, 200),
        #         "weights": Categorical(["uniform", "distance"]),
        #         "algorithm": Categorical(["auto", "ball_tree", "kd_tree", "brute"]),
        #         "metric": Categorical(["minkowski", "euclidean", "manhattan", "chebyshev"]),
        #         "p": Integer(1, 2),
        #         "leaf_size": Integer(10, 100),
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
        #         "n_estimators": Integer(100, 5000),
        #         "learning_rate": Real(1e-3, 0.5, prior="log-uniform"),
        #         "max_depth": Integer(2, 100),
        #         "min_child_weight": Integer(1, 50),
        #         "subsample": Real(0.3, 1.0),
        #         "colsample_bytree": Real(0.3, 1.0),
        #         "colsample_bylevel": Real(0.3, 1.0),
        #         "colsample_bynode": Real(0.3, 1.0),
        #         "gamma": Real(1e-10, 10.0, prior="log-uniform"),
        #         "reg_alpha": Real(1e-10, 100.0, prior="log-uniform"),
        #         "reg_lambda": Real(1e-3, 100.0, prior="log-uniform"),
        #         "max_delta_step": Integer(0, 10),
        #         "max_bin": Integer(32, 512),
        #         "max_leaves": Categorical([0, 31, 63, 127, 255, 511]),
        #         "num_parallel_tree": Integer(1, 8),
        #         "grow_policy": Categorical(["depthwise", "lossguide"]),
        #     },
        # ),
    }


def run_hyperparameter_search():
    features, target, _ = load_modeling_data()
    features, target = sample_training_rows(features, target)
    searches = build_searches()

    cv = StratifiedKFold(
        n_splits=HYPERPARAMETER_SEARCH_FOLDS,
        shuffle=True,
        random_state=RANDOM_SEED,
    )
    loaded_results = load_hyperparameter_results()
    results = {model_name: loaded_results[model_name] for model_name in searches if model_name in loaded_results}

    if results != loaded_results:
        save_json_file(HYPERPARAMETERS_FILE, results)

    print(f"Training rows: {len(features)}")
    print(f"Feature columns: {len(features.columns)}")

    for model_name, (model, search_space) in searches.items():
        search_target = target
        if model_name == "XGBClassifier":
            search_target = LabelEncoder().fit_transform(target)
        print(f"\nSearching {model_name} ({HYPERPARAMETER_SEARCH_ITERATIONS} iterations)")
        print(
            f"DeltaYStopper enabled: delta={HYPERPARAMETER_SEARCH_DELTA_Y}, "
            f"n_best={HYPERPARAMETER_SEARCH_DELTA_Y_N_BEST}"
        )

        pbar = tqdm(total=HYPERPARAMETER_SEARCH_ITERATIONS, desc=model_name)

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

        def on_step(res):
            pbar.update(1)
            pbar.set_postfix(best_score=f"{-res.fun:.4f}")
            return False

        search.fit(
            features,
            search_target,
            callback=[
                on_step,
                DeltaYStopper(
                    HYPERPARAMETER_SEARCH_DELTA_Y,
                    n_best=HYPERPARAMETER_SEARCH_DELTA_Y_N_BEST,
                ),
            ],
        )

        pbar.close()

        best_score = float(search.best_score_)
        previous_result = results.get(model_name, {})
        previous_rows = previous_result.get("training_rows")
        previous_feature_columns = previous_result.get("feature_columns")
        previous_score = (
            float(previous_result.get("best_score", float("-inf")))
            if previous_rows == len(features) and previous_feature_columns == len(features.columns)
            else float("-inf")
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
