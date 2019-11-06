#!/usr/bin/env python3

# See https://mccormickml.com/2016/04/12/googles-pretrained-word2vec-model-in-python/

import gensim


model = gensim.models.KeyedVectors.load_word2vec_format(
    "google/GoogleNews-vectors-negative300.bin", binary=True
)
