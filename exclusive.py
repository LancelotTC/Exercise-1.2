import pandas as pd
from collections import defaultdict, Counter

# Load data
df = pd.read_csv("data/stack-overflow-data.csv")

# Split tags
df['tags_list'] = df['tags'].apply(lambda x: x.split(';'))

# ----------------------------
# 1. Build vocabulary per class
# ----------------------------
class_word_counts = defaultdict(Counter)

for _, row in df.iterrows():
    words = row['post'].split()  # raw tokens (no cleaning)
    
    for tag in row['tags_list']:
        class_word_counts[tag].update(words)

# ----------------------------
# 2. Build global word set per class
# ----------------------------
class_words = {
    tag: set(counter.keys())
    for tag, counter in class_word_counts.items()
}

# ----------------------------
# 3. Find exclusive words
# ----------------------------
exclusive_words = {}

all_tags = list(class_words.keys())

for tag in all_tags:
    others = set().union(*(class_words[t] for t in all_tags if t != tag))
    
    exclusive = class_words[tag] - others
    exclusive_words[tag] = exclusive

# ----------------------------
# 4. Show results (top 20 per class by frequency)
# ----------------------------
print("\n==== EXCLUSIVE WORDS PER CLASS ====\n")

for tag in sorted(exclusive_words.keys()):
    words = exclusive_words[tag]
    
    # rank them by frequency inside the class
    ranked = class_word_counts[tag].most_common()
    ranked = [(w, c) for w, c in ranked if w in words][:20]
    
    print(f"\n--- {tag} ---")
    
    if not ranked:
        print("No exclusive words found")
    else:
        for w, c in ranked:
            print(f"{w}: {c}")