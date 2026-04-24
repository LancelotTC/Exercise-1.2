from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from generate_vectors import (
    FeatureExtraction,
    collect_feature_words,
)
from utils import average, load_dataset


OUTPUT_DIR = Path("statistics")
OUTPUT_DIR.mkdir(exist_ok=True)


def save_plot(filename: str):
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=150)
    if "agg" not in plt.get_backend().lower():
        plt.show()
    plt.close()


def load_data(posts: list):
    df = pd.DataFrame(
        [{"post": post.post, "tags": post.tags} for post in posts],
        columns=["post", "tags"],
    )
    df["post"] = df["post"].fillna("").astype(str).str.lower()
    df["tags"] = df["tags"].fillna("").astype(str).str.lower()
    df["tags_list"] = df["tags"].apply(lambda value: value.split(";"))
    df["num_tags"] = df["tags_list"].apply(len)
    return df


def print_basic_info(df: pd.DataFrame):
    print("==== BASIC INFO ====")
    print(df[["post", "tags"]].head())
    print("\nShape:", df[["post", "tags"]].shape)
    print("\nColumns:", pd.Index(["post", "tags"]))
    print("\nMissing values:\n", df[["post", "tags"]].isnull().sum())


def tag_is_in_text(text: str, tag: str):
    return FeatureExtraction.mentions_tag(text, tag) == "1"


def analyze_tag_distribution(df: pd.DataFrame):
    print("\n==== TAG EXPLORATION ====")
    print("\nAverage number of tags per question:", average(df["num_tags"]))
    print("Max number of tags:", df["num_tags"].max())

    tag_counts = Counter(tag for tags in df["tags_list"] for tag in tags if tag)

    print("\nTop 20 most common tags:")
    for tag, count in tag_counts.most_common(20):
        print(f"{tag}: {count}")

    tags, counts = zip(*tag_counts.most_common(20))
    plt.figure(figsize=(10, 5))
    plt.bar(tags, counts)
    plt.title("Top 20 Tags")
    plt.xlabel("Tag")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    save_plot("classes distribution.png")

    return tag_counts


def analyze_tag_match(df: pd.DataFrame):
    result = (
        df.assign(
            match=df.apply(
                lambda row: tag_is_in_text(row["post"], row["tags"]),
                axis=1,
            )
        )
        .groupby("tags")["match"]
        .mean()
        .sort_values(ascending=False)
    )

    print("\n==== PROPORTION OF POSTS CONTAINING THEIR TAG ====")
    print(result)

    plt.figure(figsize=(10, 5))
    result.plot(kind="bar")
    plt.title("Proportion of posts containing their tag")
    plt.xlabel("Tag")
    plt.ylabel("Proportion")
    plt.xticks(rotation=45, ha="right")
    save_plot("tag self mention proportion.png")


def analyze_wrong_tag_mentions(df: pd.DataFrame):
    rows = []

    for tag in sorted(df["tags"].unique()):
        contains_tag = df["post"].apply(lambda post: tag_is_in_text(post, tag))
        total_mentions = contains_tag.sum()

        if total_mentions == 0:
            continue

        wrong_mentions = (contains_tag & (df["tags"] != tag)).sum()
        rows.append(
            {
                "tag": tag,
                "proportion_wrong": wrong_mentions / total_mentions,
                "total_mentions": int(total_mentions),
                "wrong_mentions": int(wrong_mentions),
            }
        )

    result = pd.DataFrame(rows).sort_values("proportion_wrong", ascending=False)

    print("\n==== PROPORTION OF WRONG TAG MENTIONS ====")
    print(result)

    plt.figure(figsize=(10, 5))
    plt.bar(result["tag"], result["proportion_wrong"])
    plt.title("Proportion of posts mentioning a tag where the true tag is different")
    plt.xlabel("Mentioned tag")
    plt.ylabel("Proportion with different true tag")
    plt.xticks(rotation=45, ha="right")
    save_plot("tag mismatch proportion.png")


def analyze_top_words(top_words_by_tag: dict[str, list[str]], posts):
    tag_word_counts = {tag: Counter() for tag in top_words_by_tag}

    for post in posts:
        words = FeatureExtraction.content_words(post.post)
        for tag in post.tags.lower().split(";"):
            if tag in tag_word_counts:
                tag_word_counts[tag].update(words)

    print("\n==== MOST RECURRENT WORDS PER TOP TAG ====")

    filename_map = {
        "c#": "top_words_c_sharp.png",
        "c++": "top_words_c_plus_plus.png",
        ".net": "top_words_dot_net.png",
        "asp.net": "top_words_asp_net.png",
        "objective-c": "top_words_objective_c.png",
        "ruby-on-rails": "top_words_ruby_on_rails.png",
    }

    for tag, words in top_words_by_tag.items():
        most_common = [(word, tag_word_counts[tag][word]) for word in words]

        print(f"\n--- {tag} ---")
        for word, count in most_common:
            print(f"{word}: {count}")

        words = [word for word, _ in most_common]
        counts = [count for _, count in most_common]

        filename = filename_map.get(tag, f"top_words_{tag}.png")
        filename = filename.replace("/", "_")

        plt.figure(figsize=(8, 4))
        plt.bar(words, counts)
        plt.title(f"Top 10 words for tag: {tag}")
        plt.xlabel("Word")
        plt.ylabel("Count")
        plt.xticks(rotation=45, ha="right")
        save_plot(filename)


def analyze_exclusive_words(exclusive_words_by_tag: dict[str, list[str]], posts):
    exclusive_word_counts = {tag: Counter() for tag in exclusive_words_by_tag}

    for post in posts:
        words = FeatureExtraction.unique_words(post.post)
        for tag in post.tags.lower().split(";"):
            if tag in exclusive_word_counts:
                exclusive_word_counts[tag].update(words)

    print("\n==== EXCLUSIVE WORDS PER CLASS ====\n")

    for tag in sorted(exclusive_words_by_tag):
        ranked_words = [(word, exclusive_word_counts[tag][word]) for word in exclusive_words_by_tag[tag]]
        print(f"--- {tag} ---")
        if not ranked_words:
            print("No exclusive words found")
        else:
            for word, count in ranked_words:
                print(f"{word}: {count}")
        print()


def main():
    posts = load_dataset()
    df = load_data(posts)
    _, top_words_by_tag, exclusive_words_by_tag = collect_feature_words(posts)
    print_basic_info(df)

    analyze_tag_distribution(df)
    analyze_tag_match(df)
    analyze_wrong_tag_mentions(df)
    analyze_top_words(top_words_by_tag, posts)
    analyze_exclusive_words(exclusive_words_by_tag, posts)


if __name__ == "__main__":
    main()
