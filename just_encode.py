#!usr/bin/env python
from sklearn.preprocessing import OneHotEncoder
from sys import argv
import numpy as np

if not len(argv) == 2:
	print("Usage: python just_encode.py <arr.npy>")

print("Array loading...")
arr = np.load(argv[1])

print("Array loaded. Encoding...")
enc = OneHotEncoder()
transformed = enc.fit_transform(arr)

print("Encoding finished. Saving as encode.npy")
np.save("encoded.npy", transformed.toarray())
#import scipy
#scipy.sparse.save_npz("encoded.npz", transformed)
