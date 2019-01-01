from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imageLoader import SimpleDatasetLoader
from imageProcessor import SimplePreprocessor
from imutils import paths
import argparse
import numpy as np


# construct the argument parse and parse the arguments
#ap = argparse.ArgumentParser()
#ap.add_argument("-d", "--dataset", required=True,help="path to input dataset")
#ap.add_argument("-k", "--neighbors", type=int, default=1,help="# of nearest neighbors for classification")
#ap.add_argument("-j", "--jobs", type=int, default=-1,help="# of jobs for k-NN distance (-1 uses all available cores)")
#args = vars(ap.parse_args())



# grab the list of images that we’ll be describing
print("[INFO] loading images...")
imagePaths = list(paths.list_images("train_4000"))

# initialize the image preprocessor, load the dataset from disk,
# and reshape the data matrix
sp = SimplePreprocessor(32, 32)
sdl = SimpleDatasetLoader(preprocessors=[sp])
(data, labels) = sdl.loader(imagePaths, verbose=500)
print (labels)
print(data)
print(data.shape)


#data = data.reshape((data.shape[0], 32,32,3))

# show some information on memory consumption of the images
print("[INFO] features matrix: {:.1f}MB".format(data.nbytes / (1024 * 1000.0)))

# encode the labels as integers
le = LabelEncoder()
labels = le.fit_transform(labels)

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels,test_size=0.25, random_state=42)
print(trainX.shape[0])
#这里必须要加下面这两项，将data的形式（4000,1,3072）变成（4000,3072）
trainX = np.reshape(trainX,(trainX.shape[0],-1))
testX = np.reshape(testX,(testX.shape[0],-1))
print(trainX)
print(trainX.shape)




# train and evaluate a k-NN classifier on the raw pixel intensities
print("[INFO] evaluating k-NN classifier...")
model = KNeighborsClassifier(n_neighbors=2,n_jobs=1)
model.fit(trainX, trainY)
print(classification_report(testY, model.predict(testX),target_names=le.classes_))