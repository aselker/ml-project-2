#!/usr/bin/env python3
"""
Import and pad vectorized articles
"""

import numpy as np
import sys

assert sys.argv[1]
assert sys.argv[2]

articles = np.load(sys.argv[1], allow_pickle=True)
lengths = [len(a) for a in articles]

print("Max length:", max(lengths))
cutoff_len = int(np.quantile(lengths, 0.4))
print("Cutoff length:", cutoff_len)

articles = [a[:cutoff_len] for a in articles if len(a) > 0]

max_len = max((len(a) for a in articles))
width = len(articles[0][0])

even_articles = np.zeros((len(articles), max_len, width))
for i, article in enumerate(articles):
    even_articles[i][: len(article)] = article

np.save(sys.argv[2], even_articles)
