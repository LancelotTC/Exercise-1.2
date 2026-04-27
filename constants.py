from pathlib import Path

DATA_FOLDER = Path("data/")
DATA_FILE = DATA_FOLDER / "stack-overflow-data.csv"
VECTORS_FILE = DATA_FOLDER / "stack-overflow-vectors.csv"
ENGLISH_STOPWORDS_FILE = DATA_FOLDER / "corpora/stopwords/english"
HYPERPARAMETERS_FILE = Path("hyperparameters.json")
PREDICTIONS_DIR = Path("predictions")

VECTOR_POST_LIMIT = None
TFIDF_MAX_FEATURES = 500
TFIDF_MIN_DF = 2
TFIDF_MAX_DF = 0.95
TFIDF_NGRAM_RANGE = (1, 2)
RANDOM_SEED = 42
HYPERPARAMETER_SEARCH_ROWS = None
HYPERPARAMETER_SEARCH_ITERATIONS = 20
HYPERPARAMETER_SEARCH_FOLDS = 3
HYPERPARAMETER_SEARCH_JOBS = 11
HYPERPARAMETER_SEARCH_DELTA_Y = 5e-4
HYPERPARAMETER_SEARCH_DELTA_Y_N_BEST = 8
VALIDATION_SIZE = 0.05

ID_COLUMN = "id"
TEXT_COLUMN = "post"
TARGET_COLUMN = "tags"
TRUE_TAG_COLUMN = "true_tag"
PREDICTED_TAG_COLUMN = "predicted_tag"
PREDICTED_PROBABILITY_COLUMN = "predicted_probability"
IS_CORRECT_COLUMN = "is_correct"

PHRASE_GROUPS = {
    ".net": ("assembly",),
    "python": ("def", "import", "traceback", "pip", "django", "pandas"),
    "java": ("public class", "system.out", "jvm", "spring", "maven"),
    "javascript": ("function", "document.", "node", "npm", "promise"),
    "jquery": ("$(", ".click", ".on(", ".ajax"),
    "angularjs": ("ng-", "$scope", "$http", "directive"),
    "php": ("<?php", "$_post", "mysqli", "echo"),
    "c#": (
        "using",
        "namespace",
        "console.writeline",
        "linq",
        "console.readline();",
        "console.write(",
        "system.text;",
        "messagebox.show(",
        "get;",
        "set;",
        "///",
        "system;",
    ),
    "c++": ("std::", "cout", "endl", "template", "::", "endl;"),
    "c": ("printf", "malloc", "scanf", "#include", "pointer"),
    "objective-c": (
        "@interface",
        "@implementation",
        "nsstring",
        "[self",
        "setenv",
        "&lt;foundation/foundation.h&gt;",
        "imageview.image",
        "[[nsautoreleasepool",
        "nsobject",
        "objective",
        "161",
        "<pre><code>@interface",
        "@end",
        "@property",
    ),
    "ios": ("uiviewcontroller", "storyboard", "xcode", "uibutton"),
    "iphone": ("uiviewcontroller", "storyboard", "xcode", "uibutton"),
    "android": (
        "activity",
        "intent",
        "xml",
        "setcontentview",
        "fragment",
        "android:layout_width=",
        "android:layout_height=",
        "wrap_content",
        "android:id=",
        "match_parent",
    ),
    "sql": ("select", "join", "group by", "index"),
    "mysql": ("mysql", "sqlstate"),
    "html": ("<div", "<form", "<table", "doctype"),
    "css": ("padding", "margin", "display:", "position:", "float:"),
    "ruby-on-rails": ("activerecord", "migration", "routes", "gem", "bundle", "has_many", "rails", "rail"),
}

PHRASE_FEATURES = tuple(dict.fromkeys(phrase for phrases in PHRASE_GROUPS.values() for phrase in phrases))

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
    "ruby-on-rails": (
        "ruby on rails",
        "rubyonrails",
    ),
    "sql": (),
}
