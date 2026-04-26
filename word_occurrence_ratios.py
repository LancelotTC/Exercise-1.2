from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from generate_vectors import FeatureExtraction
from utils import load_dataset


OUTPUT_DIR = Path("statistics") / "word_occurrence_ratios"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TOP_WORDS_PER_CLASS = 30


def collect_word_counts(posts):
    word_counts_by_class = defaultdict(Counter)
    total_word_counts = Counter()

    for post in posts:
        tag = FeatureExtraction.normalize(post.tags)
        words = FeatureExtraction.content_words(post.post)
        word_counts_by_class[tag].update(words)
        total_word_counts.update(words)

    return word_counts_by_class, total_word_counts


def build_ratio_frame(word_counts_by_class, total_word_counts):
    rows = []

    for tag in sorted(word_counts_by_class):
        for word, occs_in_class in word_counts_by_class[tag].items():
            occs_in_others = total_word_counts[word] - occs_in_class
            ratio = float("inf") if occs_in_others == 0 else occs_in_class / occs_in_others
            rows.append(
                {
                    "tag": tag,
                    "word": word,
                    "occs_in_class": occs_in_class,
                    "occs_in_others": occs_in_others,
                    "ratio": ratio,
                }
            )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame

    return frame.sort_values(
        ["tag", "ratio", "occs_in_class", "word"],
        ascending=[True, False, False, True],
    )


def print_top_words(top_frame: pd.DataFrame):
    for tag, group in top_frame.groupby("tag", sort=False):
        print(f"\n=== {tag} ===")
        for row in group.itertuples(index=False):
            ratio = "inf" if row.ratio == float("inf") else f"{row.ratio:.4f}"
            print(
                f"{row.word}: occs_in_class={row.occs_in_class}, " f"occs_in_others={row.occs_in_others}, ratio={ratio}"
            )


def main():
    posts = load_dataset()
    word_counts_by_class, total_word_counts = collect_word_counts(posts)
    ratio_frame = build_ratio_frame(word_counts_by_class, total_word_counts)

    all_rows_path = OUTPUT_DIR / "word_occurrence_ratios_all.csv"
    top_rows_path = OUTPUT_DIR / "word_occurrence_ratios_top.csv"

    ratio_frame.to_csv(all_rows_path, index=False)
    top_frame = ratio_frame.groupby("tag", sort=False).head(TOP_WORDS_PER_CLASS).copy()
    top_frame.to_csv(top_rows_path, index=False)

    print_top_words(top_frame)
    print(f"\nWrote all rows to {all_rows_path}")
    print(f"Wrote top rows to {top_rows_path}")


if __name__ == "__main__":
    main()
