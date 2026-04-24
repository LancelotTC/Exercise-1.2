import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from constants import ENGLISH_STOPWORDS_FILE, VECTOR_POST_LIMIT, VECTORS_FILE
from utils import Post, load_dataset


TOP_TAG_LIMIT = 10
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
    )

    @staticmethod
    def normalize(text: str):
        return str(text or "").lower().strip()

    @staticmethod
    def words(text: str):
        return FeatureExtraction.normalize(text).split()

    @staticmethod
    def content_words(text: str):
        return [word for word in FeatureExtraction.words(text) if word not in FeatureExtraction.STOP_WORDS]

    @staticmethod
    def unique_words(text: str):
        return set(FeatureExtraction.words(text))

    @staticmethod
    def mentions_tag(text: str, tag: str):
        normalized_text = FeatureExtraction.normalize(text)
        normalized_tag = FeatureExtraction.normalize(tag)
        pattern = rf"(?<![a-z0-9#\+\.-]){re.escape(normalized_tag)}(?![a-z0-9#\+\.-])"
        return str(int(bool(re.search(pattern, normalized_text))))

    @staticmethod
    def top_words_score(text: str, words: list[str]):
        word_counts = Counter(FeatureExtraction.content_words(text))
        return str(sum(word_counts[word] for word in words))

    @staticmethod
    def exclusive_words_score(text: str, words: list[str]):
        text_words = FeatureExtraction.unique_words(text)
        return str(sum(word in text_words for word in words))


def split_tags(tags: str):
    return [tag for tag in FeatureExtraction.normalize(tags).split(";") if tag]


def feature_name(name: str):
    normalized = FeatureExtraction.normalize(name)

    if normalized in SPECIAL_FEATURE_NAMES:
        return SPECIAL_FEATURE_NAMES[normalized]

    return re.sub(r"[^a-z0-9]+", "_", normalized).strip("_")


def collect_feature_words(posts: list[Post]):
    tag_counts = Counter()
    top_word_counts = defaultdict(Counter)
    exclusive_word_counts = defaultdict(Counter)

    for post in posts:
        tags = split_tags(post.tags)
        tag_counts.update(tags)

        content_words = FeatureExtraction.content_words(post.post)
        unique_words = FeatureExtraction.unique_words(post.post)

        for tag in tags:
            top_word_counts[tag].update(content_words)
            exclusive_word_counts[tag].update(unique_words)

    all_tags = sorted(tag_counts)
    top_words_by_tag: dict[str, list[str]] = {}

    for tag, _ in tag_counts.most_common(TOP_TAG_LIMIT):
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
    top_words_by_tag: dict[str, list[str]],
    exclusive_words_by_tag: dict[str, list[str]],
):
    row: dict[str, int | str] = {
        "id": int(post.id) if post.id is not None else row_index,
        "tags": FeatureExtraction.normalize(post.tags),
    }

    for tag in all_tags:
        row[f"mentions_tag__{feature_name(tag)}"] = int(FeatureExtraction.mentions_tag(post.post, tag))

    for tag, words in top_words_by_tag.items():
        row[f"top_words_score__{feature_name(tag)}"] = int(FeatureExtraction.top_words_score(post.post, words))

    for tag in all_tags:
        row[f"exclusive_words_score__{feature_name(tag)}"] = int(
            FeatureExtraction.exclusive_words_score(post.post, exclusive_words_by_tag[tag])
        )

    return row


def write_vectors(posts: list[Post], reference_posts: list[Post], output_path: Path):
    all_tags, top_words_by_tag, exclusive_words_by_tag = collect_feature_words(reference_posts)
    rows = [
        build_vector_row(post, index, all_tags, top_words_by_tag, exclusive_words_by_tag)
        for index, post in enumerate(posts)
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)


if __name__ == "__main__":
    all_posts = load_dataset()
    posts = all_posts if VECTOR_POST_LIMIT is None else all_posts[:VECTOR_POST_LIMIT]

    write_vectors(posts, all_posts, VECTORS_FILE)
    print(f"Wrote {len(posts)} vector rows to {VECTORS_FILE}")
