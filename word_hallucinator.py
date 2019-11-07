#!/usr/bin/env python3

import sys
import gensim
import numpy as np
import torch as t
from torch import nn

from my_model import MyModel

print("Loading saved model...")
saved_data = t.load(sys.argv[1])

model = MyModel(*saved_data["params"])
model.load_state_dict(saved_data["state_dict"])


print("Loading word2vec model...")
w2v = gensim.models.KeyedVectors.load_word2vec_format(
    "google/GoogleNews-vectors-negative300.bin", binary=True
)


words = [w2v["The"]]
state = None

for _ in range(100):
    output, state = model(w2v[words[-1]], state)
    word = w2v.most_similar(positive=output)
    words.append(word)
    print(word)
