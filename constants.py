from pathlib import Path

DATA_FOLDER = Path("data/")
DATA_FILE = DATA_FOLDER / "stack-overflow-data.csv"
VECTORS_FILE = DATA_FOLDER / "stack-overflow-vectors.csv"
ENGLISH_STOPWORDS_FILE = DATA_FOLDER / "corpora/stopwords/english"
HYPERPARAMETERS_FILE = Path("hyperparameters.json")
PREDICTIONS_DIR = Path("predictions")

VECTOR_POST_LIMIT = None
RANDOM_SEED = 42
HYPERPARAMETER_SEARCH_ROWS = None
HYPERPARAMETER_SEARCH_ITERATIONS = 60
HYPERPARAMETER_SEARCH_FOLDS = 3
HYPERPARAMETER_SEARCH_JOBS = -1
VALIDATION_SIZE = 0.2

ID_COLUMN = "id"
TEXT_COLUMN = "post"
TARGET_COLUMN = "tags"
TRUE_TAG_COLUMN = "true_tag"
PREDICTED_TAG_COLUMN = "predicted_tag"
PREDICTED_PROBABILITY_COLUMN = "predicted_probability"
IS_CORRECT_COLUMN = "is_correct"

PHRASE_FEATURES = (
    # Add any phrases here to create one binary vector column per phrase.
    "public static",
    "system.out.println",
    "select * from",
    "__name__",
    "__main__",
    "decorator",
    "numpy",
    "pandas",
    "@media",
    "directive",
    "ng-",
    "<?php",
    "microsoft",
    "apple",
    "cout",
    "system.out",
    "manifest",
    "compile",
)

ALTERNATIVE_CLASS_SPELLINGS = {
    ".net": ("dotnet", "dot net"),
    "android": (),
    "angularjs": ("angular js", "angular.js"),
    "asp.net": ("aspnet", "asp net"),
    "c": (),
    "c#": ("csharp", "c sharp", "c-sharp"),
    "c++": ("cpp", "cplusplus"),
    "css": (),
    "html": (),
    "ios": (),
    "iphone": ("i phone",),
    "java": (),
    "javascript": ("java script",),
    "jquery": (),
    "mysql": ("my sql",),
    "objective-c": ("objective c", "objectivec"),
    "php": (),
    "python": (),
    "ruby-on-rails": ("ruby on rails", "rubyonrails"),
    "sql": (),
}
