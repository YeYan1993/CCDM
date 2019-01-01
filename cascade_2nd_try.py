import cv2
import numpy as np
from numpy import *
import pandas as pd
from imutils import paths
import os
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, scale
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix  # 查看预测结果
from skimage import feature as ft
from sklearn.cross_validation import cross_val_score
from skimage.feature import greycomatrix,greycoprops #灰度共生矩阵
from sklearn.feature_selection import SelectKBest,chi2 #卡方检验
import pickle  # 导入和生成模型
import datetime

startime = datetime.datetime.now()

# 导入图像数据,进行第一次利用hog特征进行训练
# imgPaths = list(paths.list_images("F:\pycharm_project\First_image_classifier\leedsbutterfly\images"))
imgPaths = list(paths.list_images("leedsbutterfly/data_bf"))
data_1 = []
labels1 = []

for (i, imgPath) in enumerate(imgPaths):
    img = cv2.imread(imgPath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # np.array.flatten()将数组变为一维
    img = cv2.resize(img, (32, 32))
    feature, hogimage = ft.hog(img, orientations=9, pixels_per_cell=(6, 6),
                               cells_per_block=(3, 3), transform_sqrt=None, visualise=True)
    label = imgPath.split(os.path.sep)[-2]
    data_1.append(feature)
    labels1.append(label)
    print("Image Loading ....")
    print("Processed {}/{}".format(i + 1, len(imgPaths)))

data1 = np.array(data_1)
labels1 = np.array(labels1)

# 将labels1 转成整形表示
le = LabelEncoder()
labels2 = le.fit_transform(labels1)
# 标准化
data1 = scale(data1)

# 将第一次训练之后的概率类别表放到full_pred1中
full_pred1 = []
model1 = SVC(C=49, kernel='rbf', gamma='auto', class_weight='balanced', probability=True)
(trainX, testX, trainY, testY) = train_test_split(data1, labels1, test_size=0.25, random_state=42)
model1.fit(trainX, trainY)
pred = model1.predict_proba(testX)  # pred返回的是10个类的概率值
full_pred1.append(pred)

# 1.对hog特征训练之后的full_pred1得到的概率值进行逐行遍历，每行选取最大的10个值，并且将每个值对应的类别标签放到一个新的矩阵中去
# 形成一个以test为行，10个label为列的label_matrix
rows = full_pred1.shape[0]
label_matrix = []

# 这里并不需要对每一行的10个概率值进行排序，只是将他们对应的类别标签进行提取
labelsof_10 = [labelsof_0,labelsof_1,labelsof_2,labelsof_3,labelsof_4,labelsof_5,labelsof_6,labelsof_7,labelsof_8,labelsof_9]

for row1 in range(rows):
    for labelsof in labelsof_10:
        for i in range(10):
            labelsof = np.argpartition(full_pred1[row1,:],-1-i)[-1-i]





    labsof1 = np.argpartition(full_pred[0][row1, :], -1)[-1]
    labsof2 = np.argpartition(full_pred[0][row1, :], -2)[-2]
    labsof3 = np.argpartition(full_pred[0][row1, :], -3)[-3]
    a.append(labsof1)
    b.append(labsof2)
    c.append(labsof3)
