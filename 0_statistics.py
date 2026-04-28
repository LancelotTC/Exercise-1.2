from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd

from constants import PHRASE_GROUPS
from utils import load_dataset, load_vector_generation_module

vector_module = load_vector_generation_module()
FeatureExtraction = vector_module.FeatureExtraction
collect_feature_words = vector_module.collect_feature_words
feature_name = vector_module.feature_name

OUTPUT_DIR = Path("statistics")
PHRASE_OUTPUT_DIR = OUTPUT_DIR / "phrase_presence"
WORD_RATIO_OUTPUT_DIR = OUTPUT_DIR / "word_occurrence_ratios"

for output_dir in (OUTPUT_DIR, PHRASE_OUTPUT_DIR, WORD_RATIO_OUTPUT_DIR):
    output_dir.mkdir(parents=True, exist_ok=True)


def save_plot(path: Path):
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    if "agg" not in plt.get_backend().lower():
        plt.show()
    plt.close()


def load_data(posts: list):
    dataframe = pd.DataFrame(
        [{"post": post.post, "tags": post.tags} for post in posts],
        columns=["post", "tags"],
    )
    dataframe["post"] = dataframe["post"].fillna("").astype(str).str.lower()
    dataframe["tags"] = dataframe["tags"].fillna("").astype(str).str.lower()
    dataframe["num_tags"] = 1
    return dataframe


def print_basic_info(dataframe: pd.DataFrame):
    print("==== BASIC INFO ====")
    print(dataframe[["post", "tags"]].head())
    print("\nShape:", dataframe[["post", "tags"]].shape)
    print("\nColumns:", pd.Index(["post", "tags"]))
    print("\nMissing values:\n", dataframe[["post", "tags"]].isnull().sum())


def tag_is_in_text(text: str, tag: str):
    return bool(FeatureExtraction.mentions_tag(text, tag))


def analyze_tag_distribution(dataframe: pd.DataFrame):
    print("\n==== TAG EXPLORATION ====")
    print("\nAverage number of tags per question:", dataframe["num_tags"].mean())
    print("Max number of tags:", dataframe["num_tags"].max())

    tag_counts = Counter(dataframe["tags"])

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
    save_plot(OUTPUT_DIR / "classes distribution.png")


def analyze_tag_match(dataframe: pd.DataFrame):
    result = (
        dataframe.assign(
            match=dataframe.apply(
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

    plt.figure(figsize=(10, 6))
    result.iloc[::-1].plot(kind="barh")
    plt.title("Proportion of posts containing their tag")
    plt.xlabel("Proportion")
    plt.ylabel("Tag")
    plt.xlim(0, 1)
    save_plot(OUTPUT_DIR / "tag self mention proportion.png")


def analyze_wrong_tag_mentions(dataframe: pd.DataFrame):
    rows = []

    for tag in sorted(dataframe["tags"].unique()):
        contains_tag = dataframe["post"].apply(lambda post: tag_is_in_text(post, tag))
        total_mentions = contains_tag.sum()

        if total_mentions == 0:
            continue

        wrong_mentions = (contains_tag & (dataframe["tags"] != tag)).sum()
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

    plot_frame = result.iloc[::-1]
    plt.figure(figsize=(10, 6))
    plt.barh(plot_frame["tag"], plot_frame["proportion_wrong"])
    plt.title("Proportion of posts mentioning a tag where the true tag is different")
    plt.xlabel("Proportion with different true tag")
    plt.ylabel("Mentioned tag")
    plt.xlim(0, 1)
    save_plot(OUTPUT_DIR / "tag mismatch proportion.png")


def analyze_top_words(top_words_by_tag: dict[str, list[str]], posts):
    tag_word_counts = {tag: Counter() for tag in top_words_by_tag}

    for post in posts:
        tag = FeatureExtraction.normalize(post.tags)
        if tag in tag_word_counts:
            tag_word_counts[tag].update(FeatureExtraction.content_words(post.post))

    print("\n==== MOST RECURRENT WORDS PER CLASS ====")

    for tag in sorted(top_words_by_tag):
        ranked_words = [(word, tag_word_counts[tag][word]) for word in top_words_by_tag[tag]]

        print(f"\n--- {tag} ---")
        for word, count in ranked_words:
            print(f"{word}: {count}")

        words = [word for word, _ in ranked_words][::-1]
        counts = [count for _, count in ranked_words][::-1]

        plt.figure(figsize=(8, 4.5))
        plt.barh(words, counts)
        plt.title(f"Top words for tag: {tag}")
        plt.xlabel("Count")
        plt.ylabel("Word")
        save_plot(OUTPUT_DIR / f"top_words_{feature_name(tag)}.png")


def analyze_exclusive_words(exclusive_words_by_tag: dict[str, list[str]], posts):
    exclusive_word_counts = {tag: Counter() for tag in exclusive_words_by_tag}

    for post in posts:
        tag = FeatureExtraction.normalize(post.tags)
        if tag in exclusive_word_counts:
            exclusive_word_counts[tag].update(FeatureExtraction.unique_words(post.post))

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


def summarize_phrase_distribution(dataframe: pd.DataFrame):
    class_sizes = dataframe["tags"].value_counts()
    rows = []

    for expected_class, phrases in PHRASE_GROUPS.items():
        for phrase in phrases:
            present_mask = dataframe["post"].map(lambda text: FeatureExtraction.count_phrase(text, phrase) > 0)
            present_rows = dataframe[present_mask]
            total_present_count = len(present_rows)
            top_classes = present_rows["tags"].value_counts().head(3)

            if top_classes.empty:
                rows.append(
                    {
                        "expected_class": expected_class,
                        "phrase": phrase,
                        "rank": 1,
                        "top_class": "",
                        "present_count": 0,
                        "total_present_count": 0,
                        "distribution_share": 0.0,
                        "class_size": 0,
                        "presence_rate_within_class": 0.0,
                    }
                )
                continue

            for rank, (top_class, present_count) in enumerate(top_classes.items(), start=1):
                class_size = int(class_sizes[top_class])
                rows.append(
                    {
                        "expected_class": expected_class,
                        "phrase": phrase,
                        "rank": rank,
                        "top_class": top_class,
                        "present_count": int(present_count),
                        "total_present_count": int(total_present_count),
                        "distribution_share": present_count / total_present_count,
                        "class_size": class_size,
                        "presence_rate_within_class": present_count / class_size,
                    }
                )

    return pd.DataFrame(rows)


def plot_phrase_group_distribution(group_summary: pd.DataFrame, expected_class: str):
    rank_colors = {
        1: "#0f766e",
        2: "#0ea5e9",
        3: "#f59e0b",
    }
    phrases = list(dict.fromkeys(group_summary["phrase"]))
    figure_height = max(4, len(phrases) * 0.65)
    figure, axis = plt.subplots(figsize=(12, figure_height))

    left = pd.Series(0.0, index=phrases)

    for rank in (1, 2, 3):
        rank_rows = group_summary[group_summary["rank"] == rank].set_index("phrase")
        shares = rank_rows["distribution_share"].reindex(phrases).fillna(0.0)
        labels = rank_rows["top_class"].reindex(phrases).fillna("")
        counts = rank_rows["present_count"].reindex(phrases).fillna(0).astype(int)

        axis.barh(
            phrases,
            shares,
            left=left,
            color=rank_colors[rank],
            edgecolor="white",
            label=f"Top {rank}",
        )

        for phrase, share in shares.items():
            if share < 0.08:
                continue

            label = labels[phrase]
            count = counts[phrase]

            if not label or not count:
                continue

            axis.text(
                left[phrase] + share / 2,
                phrase,
                f"{label}\n{count}",
                ha="center",
                va="center",
                fontsize=8,
                color="white",
            )

        left = left + shares

    other_share = (1.0 - left).clip(lower=0.0)
    if other_share.gt(0).any():
        axis.barh(
            phrases,
            other_share,
            left=left,
            color="#e5e7eb",
            edgecolor="white",
            label="Other",
        )

    axis.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    axis.set_xlim(0, 1)
    axis.set_xlabel("Share of posts containing the phrase")
    axis.set_ylabel("Phrase")
    axis.set_title(f"Top 3 class distribution for {expected_class} phrases")
    axis.grid(axis="x", alpha=0.2)
    axis.legend(loc="lower right")
    save_plot(PHRASE_OUTPUT_DIR / f"{feature_name(expected_class)}_top3_distribution.png")


def analyze_phrase_presence_distribution(dataframe: pd.DataFrame):
    print("\n==== PHRASE PRESENCE DISTRIBUTION ====")
    summary = summarize_phrase_distribution(dataframe)
    summary_path = PHRASE_OUTPUT_DIR / "phrase_presence_top3.csv"
    summary.to_csv(summary_path, index=False)

    for expected_class, group_summary in summary.groupby("expected_class", sort=False):
        plot_phrase_group_distribution(group_summary.reset_index(drop=True), expected_class)

    print(f"Wrote phrase presence summary to {summary_path}")
    print(f"Wrote phrase presence plots to {PHRASE_OUTPUT_DIR}")


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


def plot_word_ratio_groups(top_frame: pd.DataFrame):
    for tag, group in top_frame.groupby("tag", sort=False):
        group = group.reset_index(drop=True)
        labels = [word for word in group["word"]][::-1]
        values = [float(ratio) for ratio in group["ratio"]][::-1]
        ratio_labels = [f"{ratio:.2f}" for ratio in group["ratio"]][::-1]

        figure_height = max(5, len(labels) * 0.35)
        figure, axis = plt.subplots(figsize=(10, figure_height))
        bars = axis.barh(labels, values, color="#0ea5e9")
        axis.set_title(f"Top word occurrence ratios for {tag}")
        axis.set_xlabel("occs_in_class / occs_in_others")
        axis.set_ylabel("Word")
        axis.grid(axis="x", alpha=0.2)

        for bar, ratio_label in zip(bars, ratio_labels):
            axis.text(
                bar.get_width(),
                bar.get_y() + bar.get_height() / 2,
                f" {ratio_label}",
                va="center",
                ha="left",
                fontsize=8,
            )

        save_plot(WORD_RATIO_OUTPUT_DIR / f"{feature_name(tag)}_top_ratio_words.png")


def analyze_word_occurrence_ratios(posts):
    print("\n==== WORD OCCURRENCE RATIOS ====")
    word_counts_by_class, total_word_counts = collect_word_counts(posts)
    ratio_frame = build_ratio_frame(word_counts_by_class, total_word_counts)
    finite_ratio_frame = ratio_frame[ratio_frame["occs_in_others"] > 0].copy()

    all_rows_path = WORD_RATIO_OUTPUT_DIR / "word_occurrence_ratios_all.csv"
    top_rows_path = WORD_RATIO_OUTPUT_DIR / "word_occurrence_ratios_top.csv"

    ratio_frame.to_csv(all_rows_path, index=False)
    top_frame = finite_ratio_frame.groupby("tag", sort=False).head(30).copy()
    top_frame.to_csv(top_rows_path, index=False)
    plot_word_ratio_groups(top_frame)

    for tag, group in top_frame.groupby("tag", sort=False):
        print(f"\n--- {tag} ---")
        for row in group.itertuples(index=False):
            print(
                f"{row.word}: occs_in_class={row.occs_in_class}, "
                f"occs_in_others={row.occs_in_others}, ratio={row.ratio:.4f}"
            )

    print(f"\nWrote word occurrence ratios to {all_rows_path}")
    print(f"Wrote top word occurrence ratios to {top_rows_path}")
    print(f"Wrote word occurrence ratio plots to {WORD_RATIO_OUTPUT_DIR}")


def main():
    posts = load_dataset()
    dataframe = load_data(posts)
    _, top_words_by_tag, exclusive_words_by_tag = collect_feature_words(posts)

    print_basic_info(dataframe)
    analyze_tag_distribution(dataframe)
    analyze_tag_match(dataframe)
    analyze_wrong_tag_mentions(dataframe)
    analyze_top_words(top_words_by_tag, posts)
    analyze_exclusive_words(exclusive_words_by_tag, posts)
    analyze_phrase_presence_distribution(dataframe)
    analyze_word_occurrence_ratios(posts)


if __name__ == "__main__":
    main()
