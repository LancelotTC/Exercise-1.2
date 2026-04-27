# Stack Overflow Tag Classification

Classifies Stack Overflow posts into one of 20 tags from manual text features plus TF-IDF.

## Setup

### Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### Prepare the dataset

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

## Pipeline

### 0. Run statistics

```powershell
python 0_statistics.py
```

This step writes:

- class distribution plots
- tag mention plots
- top-word plots for every class
- exclusive-word summaries
- phrase presence summaries and plots
- word occurrence ratio CSVs and plots

### 1. Generate vectors

```powershell
python 1_generate_vectors.py
```

This writes `data/stack-overflow-vectors.csv`.

Current feature groups:

- `id`
- `tags`
- class mention indicators
- exclusive-word presence indicators per class
- alternative-spelling presence indicators per class
- phrase presence indicators from `PHRASE_GROUPS` in `constants.py`
- `count_special_chars`
- TF-IDF features

### 2. Search best hyperparameters

```powershell
python 2_find_best_hyperparameters.py
```

This step:

- uses `BayesSearchCV`
- uses stratified cross-validation
- uses `DeltaYStopper`
- writes results to `hyperparameters.json`

Note: the search space is defined directly in `2_find_best_hyperparameters.py`. Right now only `LogisticRegression` is enabled there.
If you change the vector schema, rerun this step before steps 3 or 4.

### 3. Run individual model predictions

```powershell
python 3_individual_predictions.py
```

This step:

- loads tuned models from `hyperparameters.json`
- uses a stratified validation split
- writes validation predictions
- writes full-dataset predictions
- writes confusion matrix CSV + PNG
- reports accuracy, macro precision, macro recall, macro specificity, macro F1

Outputs go to `predictions/<ModelName>/`.

### 4. Run ensemble predictions

```powershell
python 4_stacking_classifiers.py
```

This step combines tuned base models with:

- `StackingClassifier`
- `StackingClassifierPassthrough`
- `SoftVotingClassifier`
- `HardVotingClassifier`

Outputs go to `predictions/<EnsembleName>/`.

## Main Files

- `constants.py`: paths and runtime settings
- `utils.py`: dataset, vector, metadata, and JSON helpers
- `0_statistics.py`: analysis and plotting step
- `1_generate_vectors.py`: feature extraction and vector export
- `2_find_best_hyperparameters.py`: Bayesian hyperparameter search
- `3_individual_predictions.py`: per-model prediction step
- `4_stacking_classifiers.py`: ensemble prediction step

## Outputs

- `data/stack-overflow-vectors.csv`
- `hyperparameters.json`
- `predictions/...`
- `statistics/...`
