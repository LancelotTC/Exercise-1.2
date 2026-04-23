from tqdm import tqdm
import re, json, attrs
from pathlib import Path
from functools import partial
from collections.abc import Callable
from tools.mathtools import average


@attrs.define
class Article:
    title: str = ""
    text: str = ""
    url: str = ""

    def apply(self, func: Callable[[str], str]):
        self.title = func(self.title)
        self.text = func(self.text)

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


def statistics(articles: list[Article]):
    # [attrs.asdict(article) for article in articles],

    mean_title, max_title, min_title = (
        average(articles, lambda a: len(a.title)),
        len(max(articles, key=lambda a: len(a.title)).title),
        len(min(articles, key=lambda a: len(a.title)).title),
    )
    mean_text, max_text, min_text = (
        average(articles, lambda a: len(a.text)),
        len(max(articles, key=lambda a: len(a.text)).text),
        len(min(articles, key=lambda a: len(a.text)).text),
    )

    print(mean_title)
    print(mean_text)

    print(max_title)
    print(min_title)
    print(max_text)
    print(min_text)


if __name__ == "__main__":
    DATA_FILE = "articles_franceinfosports.json"

    with open(DATA_FILE, encoding="utf-8") as file:
        articles = [Article(**article) for article in json.load(file)]

    # articles = list(filter(lambda a: a.title.startswith("Football"), articles))

    statistics(articles)

    for article in tqdm(articles, desc="Applying operations"):
        article.apply(Operations.remove_duplicate_whitespace)
        article.apply(Operations.remove_stop_words)

        article.apply(Operations.remove_duplicate_whitespace)
        article.apply(Operations.remove_apostrophes)

        # article.apply_transformation(Operations.remove_duplicate_whitespace)
        # article.apply_transformation(Operations.lemmatize)

        article.apply(Operations.remove_duplicate_whitespace)
        article.apply(Operations.remove_double_dash)

        article.apply(Operations.remove_duplicate_whitespace)
        # This one at the end because it assumes current state is normal
        article.apply(Operations.remove_extra_whitespace)

    with open("results.json", "w", encoding="utf-8") as file:
        json.dump(
            [attrs.asdict(article) for article in articles],
            file,
        )
