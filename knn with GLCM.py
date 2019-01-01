import cv2
import numpy as np
from imutils import paths
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from skimage import feature as ft
from sklearn import preprocessing
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
from skimage.feature import greycomatrix,greycoprops
import datetime
from sklearn.svm import SVC


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
    #这里的color_feature的维度仅有8个，结果不好，用颜色特征做效果并不好
    g = greycomatrix(img,[1,2],[0,np.pi/4,np.pi/2,3*np.pi/4],symmetric=True)
    GLCM_feature = greycoprops(g).flatten()
    #这里imgPath.split(os.path.sep)[-1]是子目录下的文件名.jpg，而[-2]是子目录的名字,从而区分标签
    label = imgPath.split(os.path.sep)[-2]
    data.append(GLCM_feature)
    labels.append(label)
    print("Image Loading ....")
    print("Processed {}/{}".format(i+1,len(imgPaths)))

data1 = np.array(data)
labels1 = np.array(labels)

#规则化（预处理）
data2 = preprocessing.scale(data1)
#将labels1 转成整形表示
le = LabelEncoder()
labels2 = le.fit_transform(labels1)



#将数据集进行分类
(trainX,testX,trainY,testY) = train_test_split(data2,labels2,test_size=0.25,random_state=42)
#这里对分好之后的trainX进行交叉验证
#kf = KFold(trainX,n_folds=10)
#model = KNeighborsClassifier(p=1,n_neighbors=2,weights='uniform',algorithm='auto')
#score = cross_val_score(model,trainX,trainY,cv=10,scoring='accuracy')
#print("Accuracy is {}".format(score.mean()))

#分类器实验
# model = KNeighborsClassifier(p=1,n_neighbors=2,weights='uniform',algorithm='auto')
model = SVC(C=100, kernel = 'rbf', gamma = 'auto',class_weight = 'balanced')
model.fit(trainX,trainY)
print(classification_report(testY,model.predict(testX)))

endtime = datetime.datetime.now()
print("Using time is {}".format(endtime - starttime))
