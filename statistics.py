import pandas as pd
import numpy as np
import re
from collections import Counter
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("data/stack-overflow-data.csv")

print("==== BASIC INFO ====")
print(df.head())
print("\nShape:", df.shape)
print("\nColumns:", df.columns)
print("\nMissing values:\n", df.isnull().sum())

print("\n==== TAG EXPLORATION ====")

# Split tags into lists
df['tags_list'] = df['tags'].apply(lambda x: x.split(';'))

# Count number of tags per question
df['num_tags'] = df['tags_list'].apply(len)

print("\nAverage number of tags per question:", df['num_tags'].mean())
print("Max number of tags:", df['num_tags'].max())

# Count tag frequencies
all_tags = [tag for tags in df['tags_list'] for tag in tags]
tag_counts = Counter(all_tags)

print("\nTop 20 most common tags:")
for tag, count in tag_counts.most_common(20):
    print(f"{tag}: {count}")

# Plot top tags
top_tags = tag_counts.most_common(20)
tags, counts = zip(*top_tags)

plt.figure(figsize=(10,5))
plt.bar(tags, counts)
plt.xticks(rotation=45)
plt.title("Top 20 Tags")
plt.show()
