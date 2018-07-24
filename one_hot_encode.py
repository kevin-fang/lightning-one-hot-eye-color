from sklearn.svm import LinearSVC
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import collections
import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import os

# pull the tile data from keep
print("Pulling tile data from keep...")

tiled_data_dir = "/data-sdd/home/kfang/keep/by_id/su92l-4zz18-b8rs5x7t6gry16k/"
def get_file(name, np_file = True):
    if np_file: 
        return np.load(os.path.join(tiled_data_dir, name))
    else:
        return open(os.path.join(tiled_data_dir, name), 'r')

all_data = get_file("all.npy")
all_data += 2

nnz = np.count_nonzero(all_data, axis=0)
fracnnz = np.divide(nnz.astype(float), all_data.shape[0])

idxKeep = fracnnz >= .90
idxOP = np.arange(all_data.shape[1])
all_data = all_data[:, idxKeep]

# save data with eye color information
print("Saving eye color information...")
excludeHazel = True

# read names that have provided survey eye color data
columns = ['name', 'timestamp', 'id', 'blood_type', 'height', 'weight', 'hw_comments', 'left', 'right', 'left_desc', 'right_desc', 'eye_comments', 'hair', 'hair_desc', 'hair_comments', 'misc', 'handedness']

# pgp eye color data from survey
surveyData = pd.read_csv("./eye_color_data/PGP-Survey.csv", names=columns, na_values=['nan', '', 'NaN'])

# names of the pgp participants
surveyNames = np.asarray(surveyData['name'].values.tolist())

names_file = get_file("names.npy", np_file = False)
names = []
for line in names_file:
    names.append(line[:-1])

get_name = lambda full_name: full_name[45:53]
names = map(get_name, names)

# simple lambda function to return if the input is a string
isstr = lambda val: isinstance(val, str)

eye_color = collections.namedtuple("EyeColor", ['left', 'right'])

# lookup a name in the survey data and return a tuple of the eye colors
def getData(name, surveyData, excludeHazel=False):
    for index, row in surveyData.iterrows():
        if row['name'] == name:
            if not excludeHazel:
                return eye_color(row['left'], row['right'])
            else:
                if isstr(row['left_desc']) and isstr(row['right_desc']):
                    if 'azel' in row['left_desc'] or 'azel' in row['right_desc']:
                        return None
                return eye_color(row['left'], row['right'])

# list of tuples for index and name with eye color data (idx, name)
nameEyeMap = []
namePair = collections.namedtuple("NamePair", ['index', 'name'])

# dictionary of left and right eye colors with respective name, i.e., {"huID": 12}
leftEyeMap = {}
rightEyeMap = {}

existingNames = []

# loop through pgpNames and add eye color to maps, making sure not to add the same name twice
for i, name in enumerate(names):
    if name in surveyNames and name not in existingNames:
        existingNames.append(name)
        # change `excludeHazel=True` to include hazel in the training/testing data.
        eyeData = getData(name, surveyData, excludeHazel=excludeHazel)
        if eyeData == None:
            pass
        elif isstr(eyeData.left) and isstr(eyeData.right):
            nameEyeMap.append(namePair(i, name))
            leftEyeMap[name] = eyeData.left
            rightEyeMap[name] = eyeData.right

# create lists containing the known eye color names and the unknown eye colors.
nameIndices, correspondingNames = [], []
for pair in nameEyeMap:
    nameIndices.append(pair.index)
    correspondingNames.append(pair.name)

# convert dictionaries to lists 
leftEyeList = []
rightEyeList = []
# nametuple looks like (index, name)
for _, name in nameEyeMap:
    if isstr(leftEyeMap[name]):
        leftEyeList.append(leftEyeMap[name])
    if isstr(rightEyeMap[name]):
        rightEyeList.append(rightEyeMap[name])

blueOrNot = lambda color: 0 if int(color) > 13 else 1
leftEyeList = map(blueOrNot, leftEyeList)

# save genomes that we know the eye color of from surveys
knownData = all_data[nameIndices]
unknownData = np.delete(all_data, nameIndices, axis=0)

# save information about deleting missing/spanning data
print("Saving missing/spanning tile data")
varvals = np.full(50 * knownData.shape[1], np.nan)
nx = 0

varlist = []
for j in range(0, knownData.shape[1]):
    u = np.unique(knownData[:,j])
    varvals[nx : nx + u.size] = u
    nx = nx + u.size
    varlist.append(u)

varvals = varvals[~np.isnan(varvals)]
np.save("varvals.npy", varvals)

# one hot encode the data - fit, transform, then save for future processing
print("Encoding data...")
enc = OneHotEncoder()
transformed = enc.fit(knownData)
data = enc.transform(knownData)
encoded = data.toarray()
np.save("data_encoded.npy", encoded)
print("Finished saving data.")  

