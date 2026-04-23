import pandas as pd
from collections import Counter, defaultdict
import matplotlib.pyplot as plt

import nltk
from nltk.corpus import stopwords

# Download stopwords (run once)
nltk.download('stopwords')

# Load data
df = pd.read_csv("data/stack-overflow-data.csv")

# Split tags
df['tags_list'] = df['tags'].apply(lambda x: x.split(';'))

# Load NLTK stopwords
stop_words = set(stopwords.words('english'))

# Count words per tag
tag_word_counts = defaultdict(Counter)

for _, row in df.iterrows():
    words = [
        w for w in row['post'].split()
        if w.lower() not in stop_words
    ]
    
    tags = row['tags_list']
    
    for tag in tags:
        tag_word_counts[tag].update(words)

# Top tags
all_tags = [tag for tags in df['tags_list'] for tag in tags]
top_tags = [tag for tag, _ in Counter(all_tags).most_common(10)]

# Plot
TOP_N = 10

for tag in top_tags:
    most_common = tag_word_counts[tag].most_common(TOP_N)
    
    words = [w for w, _ in most_common]
    counts = [c for _, c in most_common]
    
    plt.figure(figsize=(8,4))
    plt.bar(words, counts)
    plt.title(f"Top {TOP_N} words for tag: {tag}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()