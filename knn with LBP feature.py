import cv2
import numpy as np
import pandas as pd
from numpy import *
from imutils import paths
import os
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from skimage import feature as ft
from sklearn import preprocessing
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest,chi2 #卡方检验
import datetime

starttime = datetime.datetime.now()

#导入图像数据
#imgPaths = list(paths.list_images("F:\pycharm_project\First_image_classifier\leedsbutterfly\images"))
imgPaths = list(paths.list_images("leedsbutterfly\data_bf"))
data = []
labels = []

for (i, imgPath) in enumerate(imgPaths):
    img = cv2.imread(imgPath)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #np.array.flatten()将数组变为一维
    img = cv2.resize(img,(32,32))
    feature = ft.local_binary_pattern(img, 12, 4, method='ror').flatten()
    # feature, hog_image = ft.hog(img, orientations=9, pixels_per_cell=(6, 6),
    #                             cells_per_block=(3, 3), transform_sqrt=None, visualise=True)
    #这里imgPath.split(os.path.sep)[-1]是子目录下的文件名.jpg，而[-2]是子目录的名字,从而区分标签
    label = imgPath.split(os.path.sep)[-2]
    data.append(feature)
    labels.append(label)
    print("Image Loading ....")
    print("Processed {}/{}".format(i+1,len(imgPaths)))


data1 = np.array(data)
labels1 = np.array(labels)

#将labels1 转成整形表示
le = LabelEncoder()
labels2 = le.fit_transform(labels1)

#进行卡方检验后效果更差，由原来的0.43减到0.35
# data2 = pd.DataFrame(data1)
# data2 = (data2 - data2.min())/(data2.max()- data2.min())
# data22 = SelectKBest(chi2, k=400).fit_transform(data2, labels2)
#归一化
data222 = StandardScaler().fit(data1)
data222 = data222.transform(data1)


#将数据集进行分类
(trainX,testX,trainY,testY) = train_test_split(data222,labels2,test_size=0.25,random_state=42)
# model = KNeighborsClassifier(p=1,n_neighbors=50,weights='uniform',algorithm='auto')
# score = cross_val_score(model,trainX,trainY,cv = 10,scoring='accuracy')
# print(score.mean())
#分类器实验
# model = KNeighborsClassifier(p=1,n_neighbors=2,weights='uniform',algorithm='auto')
model = SVC(C=10, kernel = 'rbf', gamma = 'auto',class_weight = 'balanced',probability=True)
model.fit(trainX,trainY)
print(classification_report(testY,model.predict(testX)))

endtime = datetime.datetime.now()
print("Using Time Is {}".format(endtime - starttime))