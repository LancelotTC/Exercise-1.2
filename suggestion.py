from tqdm import tqdm
import re, json, attrs
from pathlib import Path
from typing import Optional
from functools import partial
from collections.abc import Callable, Iterable
import pandas as pd


@attrs.define
class Post:
    post: str = ""
    tags: str = ""

    def apply(self, func: Callable[[str], str]):
        self.post = func(self.post)
        self.tags = func(self.tags)

    def remove_words(self, word: str):
        self.apply(partial(str.replace, old=word, new=""))


class Operations:
    @staticmethod
    def remove_extra_whitespace(string: str):
        string = string.strip()

        return re.sub(r"\s(?=\W)", "", string)

    @staticmethod
    def remove_duplicate_whitespace(string: str):
        return re.sub(r"\s{2,}", " ", string)

    @staticmethod
    def digits_to_spelled(string: str):
        from num2words import num2words

        return num2words(string)

    @staticmethod
    def spelled_to_digit(string: str):
        from word2number import w2n

        return w2n.word_to_num(string)

    @staticmethod
    def remove_stop_words(string: str):
        import nltk

        nltk.download("stopwords", quiet=True)

        STOP_WORDS = frozenset(
            Path(r"C:\Users\lance\AppData\Roaming\nltk_data\corpora\stopwords\french").read_text().split("\n")
        )

        for word in STOP_WORDS:
            string = re.sub(rf"(?<=\W)({word}|{word.upper()}|{word.capitalize()})(?=\W)", "", string)

        return string

    @staticmethod
    def remove_apostrophes(string: str):
        return string.replace("'", "").replace("’", "")

    @staticmethod
    def stem(string: str):
        from nltk.stem.snowball import SnowballStemmer

        stemmer = SnowballStemmer("french")
        return " ".join(stemmer.stem(word) for word in string.split(" "))

    @staticmethod
    def lemmatize(string: str):
        import spacy
        from unidecode import unidecode as no_accents

        lemmatizer = spacy.load("fr_core_news_md")

        return lemmatizer(no_accents(string)).text

    @staticmethod
    def remove_double_dash(string: str):
        return re.sub(r"\s?\-\-\s?", " ", string)


def average(numbers: Iterable[int | float], key: Optional[Callable] = lambda x: x) -> int | float:
    """Returns average of all numerical values in a one-dimensional Iterable or Mapping-like object"""

    if not isinstance(numbers, Iterable) or isinstance(numbers, str):
        raise TypeError(f"Expected object of type Iterable, got {type(numbers).__name__}")

    return sum((key(n) for n in numbers)) / len(numbers)


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
    DATA_FILE = "data/stack-overflow-data.csv"

    df = pd.read_csv(DATA_FILE)

    posts = [Post(**post) for post in df.to_dict(orient="records")]

    statistics(posts)

    for post in tqdm(posts, desc="Applying operations"):
        post.apply(Operations.remove_duplicate_whitespace)
        post.apply(Operations.remove_stop_words)

        post.apply(Operations.remove_duplicate_whitespace)
        post.apply(Operations.remove_apostrophes)

        # article.apply_transformation(Operations.remove_duplicate_whitespace)
        # article.apply_transformation(Operations.lemmatize)

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
