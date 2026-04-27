# Stack Overflow Tag Classification

Classifies Stack Overflow posts into one of 20 tags from manual text features plus TF-IDF.

## Pipeline

### 0. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 1. Prepare the dataset

Input file: `data/stack-overflow-data.csv`

Expected columns:

- `post`
- `tags`

Optional column:

- `id`

If `id` is missing and you want to persist it in the dataset:

```powershell
python add_dataset_ids.py
```

### 2. Generate vectors

```powershell
python generate_vectors.py
```

This writes `data/stack-overflow-vectors.csv`.

Current feature groups:

- `id`
- `tags`
- class mention indicators
- exclusive-word counts per class
- alternative-spelling counts per class
- phrase presence indicators from `PHRASE_GROUPS` in `constants.py`
- `count_special_chars`
- TF-IDF features

### 3. Search best hyperparameters

```powershell
python 1_find_best_hyperparameters.py
```

This step:

- uses `BayesSearchCV`
- uses stratified cross-validation
- uses `DeltaYStopper`
- writes results to `hyperparameters.json`

Note: the search space is defined directly in `1_find_best_hyperparameters.py`. Right now only `LogisticRegression` is enabled there.

### 4. Run individual model predictions

```powershell
python 2_individual_predictions.py
```

This step:

- loads tuned models from `hyperparameters.json`
- uses a stratified validation split
- writes validation predictions
- writes full-dataset predictions
- writes confusion matrix CSV + PNG
- reports accuracy, macro precision, macro recall, macro specificity, macro F1

Outputs go to `predictions/<ModelName>/`.

### 5. Run ensemble predictions

```powershell
python 3_stacking_predictions.py
```

This step combines tuned base models with:

- `StackingClassifierPassthrough`
- `SoftVotingClassifier`
- `HardVotingClassifier`

Outputs go to `predictions/<EnsembleName>/`.

## Side Analysis Scripts

These are not part of the main training pipeline:

- `statistics.py`: class distribution and word analysis plots
- `phrase_presence_distribution.py`: top-3 class distribution for handpicked phrases
- `word_occurrence_ratios.py`: words ranked by `occs_in_class / occs_in_others`

## Main Files

- `constants.py`: paths and runtime settings
- `utils.py`: dataset, vector, metadata, and JSON helpers
- `generate_vectors.py`: feature extraction and vector export
- `1_find_best_hyperparameters.py`: Bayesian hyperparameter search
- `2_individual_predictions.py`: per-model prediction step
- `3_stacking_predictions.py`: ensemble prediction step

## Outputs

- `data/stack-overflow-vectors.csv`
- `hyperparameters.json`
- `predictions/...`
- `statistics/...`
