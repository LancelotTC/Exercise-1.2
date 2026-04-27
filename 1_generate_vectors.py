import re
from collections import Counter, defaultdict
from pathlib import Path
import string

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from constants import (
    ALTERNATIVE_CLASS_SPELLINGS,
    ENGLISH_STOPWORDS_FILE,
    PHRASE_FEATURES,
    TFIDF_MAX_DF,
    TFIDF_MAX_FEATURES,
    TFIDF_MIN_DF,
    TFIDF_NGRAM_RANGE,
    VECTOR_POST_LIMIT,
    VECTORS_FILE,
)
from utils import Post, load_dataset


TOP_WORD_LIMIT = 10
EXCLUSIVE_WORD_LIMIT = 20

SPECIAL_FEATURE_NAMES = {
    ".net": "dot_net",
    "asp.net": "asp_net",
    "c#": "c_sharp",
    "c++": "c_plus_plus",
    "objective-c": "objective_c",
    "ruby-on-rails": "ruby_on_rails",
}


class FeatureExtraction:
    STOP_WORDS = frozenset(
        word.strip() for word in ENGLISH_STOPWORDS_FILE.read_text(encoding="utf-8").splitlines() if word.strip()
    ) | frozenset({"amp", "code", "gt", "lt", "pre"})
    TOKEN_PATTERN = re.compile(r"[a-z0-9#\+\._$:-]+")
    TFIDF_STRIP_CHARS = "".join(character for character in string.punctuation if character not in ".#+$_-")

    @staticmethod
    def normalize(text: str):
        return str(text or "").lower().strip()

    @staticmethod
    def words(text: str):
        return FeatureExtraction.TOKEN_PATTERN.findall(FeatureExtraction.normalize(text))

    @staticmethod
    def content_words(text: str):
        return [
            word
            for word in FeatureExtraction.words(text)
            if len(word) >= 2 and word not in FeatureExtraction.STOP_WORDS
        ]

    @staticmethod
    def unique_words(text: str):
        return set(FeatureExtraction.content_words(text))

    @staticmethod
    def tfidf_words(text: str):
        tokens = []

        for word in FeatureExtraction.words(text):
            token = word.strip(FeatureExtraction.TFIDF_STRIP_CHARS)
            if len(token) < 2 or token in FeatureExtraction.STOP_WORDS:
                continue
            if not any(character.isalnum() for character in token):
                continue
            tokens.append(token)

        return tokens

    @staticmethod
    def mentions_tag(text: str, tag: str):
        normalized_text = FeatureExtraction.normalize(text)
        normalized_tag = FeatureExtraction.normalize(tag)
        pattern = rf"(?<![a-z0-9#\+\.-]){re.escape(normalized_tag)}(?![a-z0-9#\+\.-])"
        return int(bool(re.search(pattern, normalized_text)))

    @staticmethod
    def contains_exclusive_words(text: str, words: list[str]):
        text_words = FeatureExtraction.unique_words(text)
        return int(any(word in text_words for word in words))

    @staticmethod
    def contains_alternative_spelling(text: str, spellings: tuple[str, ...]):
        return int(any(FeatureExtraction.mentions_tag(text, spelling) for spelling in spellings))

    @staticmethod
    def count_special_chars(text: str):
        normalized_text = FeatureExtraction.normalize(text)
        return sum(character in string.punctuation for character in normalized_text)

    @staticmethod
    def contains_phrase(text: str, phrase: str):
        normalized_text = FeatureExtraction.normalize(text)
        normalized_phrase = FeatureExtraction.normalize(phrase)

        if not normalized_phrase:
            return 0

        starts_with_word = normalized_phrase[0].isalnum()
        ends_with_word = normalized_phrase[-1].isalnum()
        left_boundary = r"(?<![a-z0-9])" if starts_with_word else ""
        right_boundary = r"(?![a-z0-9])" if ends_with_word else ""
        pattern = f"{left_boundary}{re.escape(normalized_phrase)}{right_boundary}"
        return int(bool(re.search(pattern, normalized_text)))


def feature_name(name: str):
    normalized = FeatureExtraction.normalize(name)

    if normalized in SPECIAL_FEATURE_NAMES:
        return SPECIAL_FEATURE_NAMES[normalized]

    return re.sub(r"[^a-z0-9]+", "_", normalized).strip("_")


def unique_column_names(prefix: str, raw_names):
    used = Counter()
    columns = []

    for raw_name in raw_names:
        base_name = f"{prefix}{feature_name(raw_name)}"
        used[base_name] += 1
        columns.append(base_name if used[base_name] == 1 else f"{base_name}_{used[base_name]}")

    return columns


def collect_feature_words(posts: list[Post]):
    tag_counts = Counter()
    top_word_counts = defaultdict(Counter)
    exclusive_word_counts = defaultdict(Counter)

    for post in posts:
        tag = FeatureExtraction.normalize(post.tags)
        tag_counts[tag] += 1

        content_words = FeatureExtraction.content_words(post.post)
        unique_words = FeatureExtraction.unique_words(post.post)

        top_word_counts[tag].update(content_words)
        exclusive_word_counts[tag].update(unique_words)

    all_tags = sorted(tag_counts)
    top_words_by_tag: dict[str, list[str]] = {}

    for tag in all_tags:
        top_words_by_tag[tag] = [word for word, _ in top_word_counts[tag].most_common(TOP_WORD_LIMIT)]

    class_words = {tag: set(word_counts) for tag, word_counts in exclusive_word_counts.items()}
    exclusive_words_by_tag: dict[str, list[str]] = {}

    for tag in all_tags:
        other_words = set().union(*(class_words[other] for other in all_tags if other != tag))
        exclusive_words = class_words[tag] - other_words
        exclusive_words_by_tag[tag] = [
            word for word, _ in exclusive_word_counts[tag].most_common() if word in exclusive_words
        ][:EXCLUSIVE_WORD_LIMIT]

    return all_tags, top_words_by_tag, exclusive_words_by_tag


def build_vector_row(
    post: Post,
    row_index: int,
    all_tags: list[str],
    exclusive_words_by_tag: dict[str, list[str]],
):
    row: dict[str, int | str] = {
        "id": int(post.id) if post.id is not None else row_index,
        "tags": FeatureExtraction.normalize(post.tags),
        "count_special_chars": FeatureExtraction.count_special_chars(post.post),
    }

    for tag in all_tags:
        row[f"contains_class__{feature_name(tag)}"] = FeatureExtraction.mentions_tag(post.post, tag)

        row[f"contains_exclusive_words__{feature_name(tag)}"] = FeatureExtraction.contains_exclusive_words(
            post.post, exclusive_words_by_tag[tag]
        )

        row[f"contains_alternative_spelling__{feature_name(tag)}"] = FeatureExtraction.contains_alternative_spelling(
            post.post, ALTERNATIVE_CLASS_SPELLINGS[tag]
        )

    for phrase in PHRASE_FEATURES:
        row[f"contains_phrase__{feature_name(phrase)}"] = FeatureExtraction.contains_phrase(post.post, phrase)

    return row


def build_tfidf_frame(reference_posts: list[Post], posts: list[Post]):
    vectorizer = TfidfVectorizer(
        tokenizer=FeatureExtraction.tfidf_words,
        token_pattern=None,
        lowercase=False,
        max_features=TFIDF_MAX_FEATURES,
        min_df=TFIDF_MIN_DF,
        max_df=TFIDF_MAX_DF,
        ngram_range=TFIDF_NGRAM_RANGE,
        sublinear_tf=True,
    )
    vectorizer.fit([post.post for post in reference_posts])
    matrix = vectorizer.transform([post.post for post in posts])
    columns = unique_column_names("tfidf__", vectorizer.get_feature_names_out())
    return pd.DataFrame(matrix.toarray(), columns=columns, dtype="float32")


def write_vectors(posts: list[Post], reference_posts: list[Post], output_path: Path):
    all_tags, _, exclusive_words_by_tag = collect_feature_words(reference_posts)
    manual_rows = [build_vector_row(post, index, all_tags, exclusive_words_by_tag) for index, post in enumerate(posts)]
    manual_frame = pd.DataFrame(manual_rows)
    tfidf_frame = build_tfidf_frame(reference_posts, posts)
    output_frame = pd.concat([manual_frame, tfidf_frame], axis=1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_frame.to_csv(output_path, index=False)


if __name__ == "__main__":
    all_posts = load_dataset()
    posts = all_posts if VECTOR_POST_LIMIT is None else all_posts[:VECTOR_POST_LIMIT]

    write_vectors(posts, all_posts, VECTORS_FILE)
    print(f"Wrote {len(posts)} vector rows to {VECTORS_FILE}")
