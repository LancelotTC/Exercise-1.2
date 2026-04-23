import pandas as pd
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
from constants import DATA_FILE
from utils import average, load_dataset

OUTPUT_DIR = Path("statistics")
OUTPUT_DIR.mkdir(exist_ok=True)


def save_plot(filename: str) -> None:
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=150)
    if "agg" not in plt.get_backend().lower():
        plt.show()
    plt.close()


def load_data() -> pd.DataFrame:
    posts = load_dataset()
    df = pd.DataFrame(
        [{"post": post.post, "tags": post.tags} for post in posts],
        columns=["post", "tags"],
    )

    if df.empty:
        df = pd.read_csv(DATA_FILE)

    df["post"] = df["post"].fillna("").astype(str).str.lower()
    df["tags"] = df["tags"].fillna("").astype(str).str.lower()
    return df


def add_tag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["tags_list"] = df["tags"].apply(lambda value: value.split(";"))
    df["num_tags"] = df["tags_list"].apply(len)
    return df


def print_basic_info(df: pd.DataFrame) -> None:
    print("==== BASIC INFO ====")
    print(df.head())
    print("\nShape:", df.shape)
    print("\nColumns:", df.columns)
    print("\nMissing values:\n", df.isnull().sum())


def compute_tag_counts(df: pd.DataFrame) -> Counter:
    all_tags = [tag for tags in df["tags_list"] for tag in tags if tag]
    return Counter(all_tags)


def print_tag_summary(df: pd.DataFrame, tag_counts: Counter) -> None:
    print("\n==== TAG EXPLORATION ====")
    print("\nAverage number of tags per question:", average(df["num_tags"]))
    print("Max number of tags:", df["num_tags"].max())

    print("\nTop 20 most common tags:")
    for tag, count in tag_counts.most_common(20):
        print(f"{tag}: {count}")


def plot_top_tags(tag_counts: Counter) -> None:
    top_tags = tag_counts.most_common(20)
    tags, counts = zip(*top_tags)

    plt.figure(figsize=(10, 5))
    plt.bar(tags, counts)
    plt.title("Top 20 Tags")
    plt.xlabel("Tag")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    save_plot("classes distribution.png")


def compute_tag_match_result(df: pd.DataFrame) -> pd.Series:
    return (
        df.assign(match=df.apply(lambda row: row["tags"] in row["post"], axis=1))
        .groupby("tags")["match"]
        .mean()
        .sort_values(ascending=False)
    )


def print_tag_match_result(result: pd.Series) -> None:
    print("\n==== PROPORTION OF POSTS CONTAINING THEIR TAG ====")
    print(result)


def plot_tag_match_result(result: pd.Series) -> None:
    plt.figure(figsize=(10, 5))
    result.plot(kind="bar")
    plt.title("Proportion of posts containing their tag")
    plt.xlabel("Tag")
    plt.ylabel("Proportion")
    plt.xticks(rotation=45, ha="right")
    save_plot("tag self mention proportion.png")


def compute_wrong_tag_result(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for tag in sorted(df["tags"].unique()):
        contains_tag = df["post"].str.contains(tag, regex=False, na=False)
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

    return pd.DataFrame(rows).sort_values("proportion_wrong", ascending=False)


def print_wrong_tag_result(result: pd.DataFrame) -> None:
    print("\n==== PROPORTION OF WRONG TAG MENTIONS ====")
    print(result)


def plot_wrong_tag_result(result: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 5))
    plt.bar(result["tag"], result["proportion_wrong"])
    plt.title("Proportion of posts mentioning a tag where the true tag is different")
    plt.xlabel("Mentioned tag")
    plt.ylabel("Proportion with different true tag")
    plt.xticks(rotation=45, ha="right")
    save_plot("tag mismatch proportion.png")


def main() -> None:
    df = load_data()
    print_basic_info(df)

    enriched_df = add_tag_features(df)

    tag_counts = compute_tag_counts(enriched_df)
    print_tag_summary(enriched_df, tag_counts)
    plot_top_tags(tag_counts)

    tag_match_result = compute_tag_match_result(df)
    print_tag_match_result(tag_match_result)
    plot_tag_match_result(tag_match_result)

    wrong_tag_result = compute_wrong_tag_result(df)
    print_wrong_tag_result(wrong_tag_result)
    plot_wrong_tag_result(wrong_tag_result)


if __name__ == "__main__":
    main()
