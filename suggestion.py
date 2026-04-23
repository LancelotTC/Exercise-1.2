import spacy
from tqdm import tqdm
import re, json, attrs
from constants import *
from utils import average, load_dataset, Post
from unidecode import unidecode as no_accents
from nltk.stem.snowball import SnowballStemmer


class Operations:
    @staticmethod
    def remove_extra_whitespace(string: str):
        string = string.strip()

        return re.sub(r"\s(?=\W)", "", string)

    @staticmethod
    def remove_duplicate_whitespace(string: str):
        return re.sub(r"\s{2,}", " ", string)

    @staticmethod
    def remove_stop_words(string: str):

        STOP_WORDS = frozenset((DATA_FOLDER / "corpora/stopwords/french").read_text().split("\n"))

        for word in STOP_WORDS:
            # string = re.sub(rf"(?<=\W)({word}|{word.upper()}|{word.capitalize()})(?=\W)", "", string)
            string = string.replace(word, "").replace(word.upper(), "").replace(word.capitalize(), "")

        return string

    @staticmethod
    def remove_apostrophes(string: str):
        return string.replace("'", "").replace("’", "")

    @staticmethod
    def stem(string: str):

        stemmer = SnowballStemmer("french")
        return " ".join(stemmer.stem(word) for word in string.split(" "))

    @staticmethod
    def lemmatize(string: str):

        lemmatizer = spacy.load("fr_core_news_md")

        return lemmatizer(no_accents(string)).text

    @staticmethod
    def remove_double_dash(string: str):
        return re.sub(r"\s?\-\-\s?", " ", string)


def statistics(articles: list[Post]):
    mean_title, max_title, min_title = (
        average(articles, lambda a: len(a.post)),
        len(max(articles, key=lambda a: len(a.post)).post),
        len(min(articles, key=lambda a: len(a.post)).post),
    )
    mean_text, max_text, min_text = (
        average(articles, lambda a: len(a.tags)),
        len(max(articles, key=lambda a: len(a.tags)).tags),
        len(min(articles, key=lambda a: len(a.tags)).tags),
    )

    print(mean_title)
    print(mean_text)

    print(max_title)
    print(min_title)
    print(max_text)
    print(min_text)


if __name__ == "__main__":
    posts = load_dataset()

    statistics(posts)

    for post in tqdm(posts, desc="Applying operations"):
        post.apply(Operations.remove_duplicate_whitespace)
        post.apply(Operations.remove_stop_words)

        post.apply(Operations.remove_duplicate_whitespace)
        post.apply(Operations.remove_apostrophes)

        post.apply(Operations.remove_duplicate_whitespace)
        post.apply(Operations.remove_double_dash)

        post.apply(Operations.remove_duplicate_whitespace)
        # This one at the end because it assumes current state is normal
        post.apply(Operations.remove_extra_whitespace)

    with open("results.json", "w", encoding="utf-8") as file:
        json.dump(
            [attrs.asdict(post) for post in posts],
            file,
        )
