#!/usr/bin/env python3

# Copied from a notebook by the ML profs

import pandas as pd
import gdown
from sklearn.feature_extraction.text import CountVectorizer
import nltk

if not os.path.exists("articles1.csv"):
    gdown.download(
        "https://drive.google.com/uc?authuser=0&id=1T8V87Hdz2IvhKjzwzKyLWA4vI6sA2wTX&export=download",
        "articles1.csv",
        quiet=False,
    )
df = pd.read_csv("articles1.csv")

# NLTK has a built-in module for extracting words from text.
# This takes a few minutes to run, so be patient.

nltk.download("punkt")


def remove_punctuation(article):
    # substitute in a regular apostrophe for '’' to work with word_tokenize
    article = article.replace("’", "'")
    tokens = nltk.tokenize.word_tokenize(article)
    words = list(filter(lambda w: any(x.isalpha() for x in w), tokens))
    return " ".join(words)


df["content_no_punctuation"] = df["content"].map(remove_punctuation)


vectorizer = CountVectorizer(
    binary=True, ngram_range=(1, 2), min_df=20, tokenizer=lambda x: x.split(" ")
)
X = vectorizer.fit_transform(df["content_no_punctuation"])
