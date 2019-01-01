import cv2
import numpy as np
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
from skimage.feature import local_binary_pattern
import datetime

startime = datetime.datetime.now()

#导入图像数据
#imgPaths = list(paths.list_images("F:\pycharm_project\First_image_classifier\leedsbutterfly\images"))
imgPaths = list(paths.list_images("leedsbutterfly/data_bf"))
data = []
labels = []

for (i, imgPath) in enumerate(imgPaths):
    img = cv2.imread(imgPath)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #np.array.flatten()将数组变为一维
    img = cv2.resize(img,(32,32))
    #feature = local_binary_pattern(img,8,8,method='ror').flatten()#适合于knn的lbp参数
    feature = local_binary_pattern(img, 32, 4, method='ror').flatten()
    #这里imgPath.split(os.path.sep)[-1]是子目录下的文件名.jpg，而[-2]是子目录的名字,从而区分标签
    label = imgPath.split(os.path.sep)[-2]
    data.append(feature)
    labels.append(label)
    print("Image Loading ....")
    print("Processed {}/{}".format(i+1,len(imgPaths)))


data1 = np.array(data)
labels1 = np.array(labels)

#规则化
data2 = preprocessing.scale(data1)
#将labels1 转成整形表示
le = LabelEncoder()
labels2 = le.fit_transform(labels1)



#将数据集进行分类
(trainX,testX,trainY,testY) = train_test_split(data2,labels2,test_size=0.25,random_state=42)
#model = SVC(C=10, kernel = 'rbf', gamma = 'auto',class_weight = 'balanced')
#score = cross_val_score(model,trainX,trainY,cv = 10,scoring='accuracy')
#print(score.mean())
#分类器实验
model = SVC(C=10, kernel = 'rbf', gamma = 'auto',class_weight = 'balanced')
#model = KNeighborsClassifier(n_neighbors=2,n_jobs=1)
model.fit(trainX,trainY)
print(classification_report(testY,model.predict(testX)))

endtime = datetime.datetime.now()
print("Using Time Is {}".format(endtime - startime))