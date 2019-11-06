#!/usr/bin/env python3

# See https://mccormickml.com/2016/04/12/googles-pretrained-word2vec-model-in-python/

import gensim
import numpy as np
import csv
import sys

assert sys.argv[1]
assert sys.argv[2]

# First, read the CSV file and tokenize the articles' bodies
# Expand CSV reader maximum field size
csv.field_size_limit(sys.maxsize)

articles = []
with open(sys.argv[1]) as f:
    c = csv.reader(f, delimiter=",")
    next(c)  # Throw away header
    for row in c:
        articles_text.append(gensim.utils.simple_preprocess(row[9]))


# Next, word2vec the articles

model = gensim.models.KeyedVectors.load_word2vec_format(
    "google/GoogleNews-vectors-negative300.bin", binary=True
)

articles_vecs = []
for i, article in enumerate(articles_text):
    vecs = [model[word] for word in article]
    articles_vecs.append(np.array(vecs))

    if i % 100:
        print("Vectorized article {}".format(i))

articles_vecs = np.asarray(articles_vecs)
np.save(sys.argv[2], articles_vecs)
