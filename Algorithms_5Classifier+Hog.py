#5种不同分类器并联针对同一feature进行实验
#classifier：KNN、SVM、RF、LR、Adaboost
#feature：Hog


import cv2
import numpy as np
import pandas as pd
from imutils import paths
import os
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder,scale
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix #查看预测结果
from skimage import feature as ft
from sklearn.cross_validation import cross_val_score
import pickle #导入和生成模型
import datetime

startime = datetime.datetime.now()

#导入图像数据
#imgPaths = list(paths.list_images("F:\pycharm_project\First_image_classifier\leedsbutterfly\images"))
imgPaths = list(paths.list_images("data_bf_500"))
data_1 = []
data_2 = []
labels = []

for (i, imgPath) in enumerate(imgPaths):
    img = cv2.imread(imgPath)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #np.array.flatten()将数组变为一维
    img = cv2.resize(img,(32,32))
    feature,hogimage= ft.hog(img, orientations=9, pixels_per_cell=(6, 6),
                                cells_per_block=(3, 3), transform_sqrt=None, visualise=True)
    #这里imgPath.split(os.path.sep)[-1]是子目录下的文件名.jpg，而[-2]是子目录的名字,从而区分标签
    label = imgPath.split(os.path.sep)[-2]
    data_1.append(feature)
    labels.append(label)
    print("Image Loading ....")
    print("Processed {}/{}".format(i+1,len(imgPaths)))


data1 = np.array(data_1)
labels1 = np.array(labels)

#规则化
data1 = scale(data1)

#将labels1 转成整形表示
le = LabelEncoder()
labels2 = le.fit_transform(labels1)


#定义分类器
model1 = SVC(C=10, kernel = 'rbf', gamma = 'auto',class_weight = 'balanced',probability=True)
model2 = KNeighborsClassifier(p=1, n_neighbors=10, weights='uniform', algorithm='auto')
model3 = RandomForestClassifier(random_state=1, n_estimators=200, min_samples_split=4, min_samples_leaf=2)
model4 = LogisticRegression(random_state=1)
model5 = AdaBoostClassifier()
algorithms = [model1,model2,model3,model4,model5]
full_pred = []

#将数据分类
(trainX,testX,trainY,testY) = train_test_split(data1,labels2,test_size=0.25,random_state=42)

#遍历整个测试集，输出概率值
for agl in algorithms:
    agl.fit(trainX,trainY)
    #这里直接将每个模型中预测的值取出来10个概率值
    #pre1 = agl.predict_proba(testX)[:,1] 这里打印的是将10个列中第1+1=2列打印出来
    pred = agl.predict_proba(testX)#pred返回的是10个类的概率值
    full_pred.append(pred)


#1.对5个（full_pred[i]）得到的概率值进行逐行遍历，每行选取最大的类标签，并将它们存在一行中，形成shape(125,5)的最大值矩阵
rows = full_pred[0].shape[0]
#做一个大循环每行进行遍历，
all = []#填充所有值
for i in range(5):
    for row in range(rows):
        labs = []
        lab_max = np.argpartition(full_pred[i][row,:],-1)[-1]
        labs.append(lab_max)
        all.append(labs)

all_lab_max = np.array(all).reshape((125,5))


#2.将得到的类别标签矩阵进行逐行统计，得到每一行的出现最多的标签作为输出值
predict_labs = []
for row2 in range(rows):
    if all_lab_max[row2,0] == all_lab_max[row2,1] == all_lab_max[row2,2] == all_lab_max[row2,3] == all_lab_max[row2,4]:
        predict_labs.append(all_lab_max[row2, 0])
    else:
        lab_row = all_lab_max[row2,:].tolist() #这里要将数组改成列表形式才能用.count
        final_max_lab = max(lab_row, key=lab_row.count)
        predict_labs.append(final_max_lab)
predict_labs = np.array(predict_labs)


#4.将预测标签与testY进行对比,通过混淆矩阵看
pt = confusion_matrix(testY,predict_labs)
print(pt)
print(classification_report(testY,predict_labs))



endtime = datetime.datetime.now()
print("Using Time Is {}".format(endtime - startime))