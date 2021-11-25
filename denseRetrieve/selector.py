import numpy as np
import faiss
import os
import re
import warnings



def rerank():
    path_to_encoding_dir = 'denseRetrieve/data/ct21/ance_encoding/'

    d = 768
    index = faiss.IndexFlatIP(d)

    index.add(faiss.randn((100, d), 345))
    index.add(faiss.randn((100, d), 678))
    recons = index.reconstruct(0)
    print(recons)