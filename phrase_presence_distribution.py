from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd

from constants import PHRASE_GROUPS, TARGET_COLUMN, TEXT_COLUMN
from generate_vectors import FeatureExtraction, feature_name
from utils import load_posts_frame


OUTPUT_DIR = Path("statistics") / "phrase_presence"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANK_COLORS = {
    1: "#0f766e",
    2: "#0ea5e9",
    3: "#f59e0b",
}


def load_data():
    dataframe = load_posts_frame()[[TEXT_COLUMN, TARGET_COLUMN]].copy()
    dataframe[TEXT_COLUMN] = dataframe[TEXT_COLUMN].fillna("").map(FeatureExtraction.normalize)
    dataframe[TARGET_COLUMN] = dataframe[TARGET_COLUMN].fillna("").map(FeatureExtraction.normalize)
    return dataframe


def summarize_phrase_distribution(dataframe: pd.DataFrame):
    class_sizes = dataframe[TARGET_COLUMN].value_counts()
    rows = []

    for expected_class, phrases in PHRASE_GROUPS.items():
        for phrase in phrases:
            present_mask = dataframe[TEXT_COLUMN].map(lambda text: FeatureExtraction.contains_phrase(text, phrase) == 1)
            present_rows = dataframe[present_mask]
            total_present_count = len(present_rows)
            top_classes = present_rows[TARGET_COLUMN].value_counts().head(3)

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


def plot_group_distribution(group_summary: pd.DataFrame, expected_class: str):
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
            color=RANK_COLORS[rank],
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
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / f"{feature_name(expected_class)}_top3_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(figure)


def main():
    dataframe = load_data()
    summary = summarize_phrase_distribution(dataframe)
    summary_path = OUTPUT_DIR / "phrase_presence_top3.csv"
    summary.to_csv(summary_path, index=False)

    for expected_class, group_summary in summary.groupby("expected_class", sort=False):
        plot_group_distribution(group_summary.reset_index(drop=True), expected_class)

    print(f"Wrote summary to {summary_path}")
    print(f"Wrote plots to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
