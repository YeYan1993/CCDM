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

# 导入图像数据
# imgPaths = list(paths.list_images("F:\pycharm_project\First_image_classifier\leedsbutterfly\images"))
imgPaths = list(paths.list_images("leedsbutterfly/data_bf"))
data_1 = []
data_2 = []
labels = []

for (i, imgPath) in enumerate(imgPaths):
    img = cv2.imread(imgPath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # np.array.flatten()将数组变为一维
    img = cv2.resize(img, (32, 32))
    feature, hogimage = ft.hog(img, orientations=9, pixels_per_cell=(6, 6),
                               cells_per_block=(3, 3), transform_sqrt=None, visualise=True)
    feature2 = ft.local_binary_pattern(img, 32, 4, method='ror').flatten()#用LBP作为特征2完全无法做到与hog进行互补，采用混合参数调节
    # g = greycomatrix(img, [1, 2, 3, 4], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], symmetric=True)
    # feature2 = greycoprops(g).flatten() #用灰度共生矩阵效果只有0.4左右
    # 这里imgPath.split(os.path.sep)[-1]是子目录下的文件名.jpg，而[-2]是子目录的名字,从而区分标签
    label = imgPath.split(os.path.sep)[-2]
    data_1.append(feature)
    data_2.append(feature2)
    labels.append(label)
    print("Image Loading ....")
    print("Processed {}/{}".format(i + 1, len(imgPaths)))

data1 = np.array(data_1)
data2 = np.array(data_2)
labels1 = np.array(labels)

# 将labels1 转成整形表示
le = LabelEncoder()
labels2 = le.fit_transform(labels1)


#选择K个最好的特征，返回选择特征后的数据
data1 = SelectKBest(chi2, k=400).fit_transform(data1, labels2)
#lbp进行矩阵非负化后，经过卡房检验得到值更低0.39
# data2 = pd.DataFrame(data2)
# data222 = (data2 - data2.min())/(data2.max()- data2.min())
# data1 = SelectKBest(chi2, k=400).fit_transform(data222, labels2)
# 标准化
data1 = scale(data1)
data2 = scale(data2)


# 定义分类器
# model1 = SVC(C=49, kernel='rbf', gamma='auto', class_weight='balanced', probability=True)#阈值为0.4时候，达到0.65（这里是不要hog进行卡方检验的）
model1 =KNeighborsClassifier(p=1, n_neighbors=92, weights='uniform', algorithm='auto') #达到0.57
# model1 = RandomForestClassifier(random_state=1, n_estimators=100, min_samples_split=4, min_samples_leaf=2)
feature_all = [data1,data2]
full_pred = []
#得到每一个feature的10类的概率值放到full_pred中
for fea in feature_all:
    (trainX, testX, trainY, testY) = train_test_split(fea, labels2, test_size=0.25, random_state=42)
    model1.fit(trainX,trainY)
    pred = model1.predict_proba(testX)  # pred返回的是10个类的概率值
    full_pred.append(pred)

# 1.对feature1（full_pred[0]）得到的概率值进行逐行遍历，每行选取最大的三个值
rows = full_pred[0].shape[0]

# 做一个大循环每行进行遍历，
a = []  # 这里是填充最大值
b = []  # 第二大
c = []  # 第三大
for row1 in range(rows):
    labsof1 = np.argpartition(full_pred[0][row1, :], -1)[-1]
    labsof2 = np.argpartition(full_pred[0][row1, :], -2)[-2]
    labsof3 = np.argpartition(full_pred[0][row1, :], -3)[-3]
    a.append(labsof1)
    b.append(labsof2)
    c.append(labsof3)


# 2.当得到的三个类别标签放到KNN（full_pred[1]）中，if 三个类都相等的时候，输出SVM的三个类中最大值的类作为预测标签
predict_labs = []
for row2 in range(rows):
    #这里应该是选取每一行的三个类别标签的最大值，而不是所有的最大值：求取三个值的最大值max_num
    a1 = full_pred[1][row2,a[row2]]
    a2 = full_pred[1][row2,b[row2]]
    a3 = full_pred[1][row2,c[row2]]
    if a1>a2:
        if a1>a3:
            max_num = a[row2]
        else:
            max_num = c[row2]
    else:
        if a2>a3:
            max_num = b[row2]
        else:
            max_num = c[row2]
    # 3.如果第一个概率矩阵中三个值中最大值超过阈值0.4，就直接选取三个里面最大值
    if full_pred[0][row2,a[row2]]>0.4:
        predict_labs.append(a[row2])  # 这里a不在循环中，且是一个一维数组，所以需要a[row2]
    # 4.else 选取这三个类中最大值对应的类最为预测标签
    else:
        predict_labs.append(max_num)  # 这里由于是d在循环中，因此不需要d[row2]

# 5.将预测标签与testY进行对比,通过混淆矩阵看
pt = confusion_matrix(testY, predict_labs)
print(pt)
print(classification_report(testY, predict_labs))

endtime = datetime.datetime.now()
print("Using Time Is {}".format(endtime - startime))