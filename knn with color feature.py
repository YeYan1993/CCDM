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
from sklearn.cross_validation import cross_val_score
from sklearn.svm import SVC
import datetime

starttime = datetime.datetime.now()

#定义颜色矩
def color_moments(img):
    if img is None:
        return
    # Convert BGR to HSV colorspace
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Split the channels - h,s,v
    h, s, v = cv2.split(hsv)
    # Initialize the color feature
    color_feature = []
    # N = h.shape[0] * h.shape[1]
    # The first central moment - average
    h_mean = np.mean(h)  # np.sum(h)/float(N)
    s_mean = np.mean(s)  # np.sum(s)/float(N)
    v_mean = np.mean(v)  # np.sum(v)/float(N)
    color_feature.extend([h_mean, s_mean, v_mean])
    # The second central moment - standard deviation
    h_std = np.std(h)  # np.sqrt(np.mean(abs(h - h.mean())**2))
    s_std = np.std(s)  # np.sqrt(np.mean(abs(s - s.mean())**2))
    v_std = np.std(v)  # np.sqrt(np.mean(abs(v - v.mean())**2))
    color_feature.extend([h_std, s_std, v_std])
    # The third central moment - the third root of the skewness
    h_skewness = np.mean(abs(h - h.mean())**3)
    s_skewness = np.mean(abs(s - s.mean())**3)
    v_skewness = np.mean(abs(v - v.mean())**3)
    h_thirdMoment = h_skewness**(1./3)
    s_thirdMoment = s_skewness**(1./3)
    v_thirdMoment = v_skewness**(1./3)
    color_feature.extend([h_thirdMoment, s_thirdMoment, v_thirdMoment])

    return color_feature

#导入图像数据
#imgPaths = list(paths.list_images("F:\pycharm_project\First_image_classifier\leedsbutterfly\images"))
imgPaths = list(paths.list_images("leedsbutterfly\data_bf"))
data = []
labels = []

for (i, imgPath) in enumerate(imgPaths):
    img = cv2.imread(imgPath)
    #img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #np.array.flatten()将数组变为一维
    img = cv2.resize(img,(32,32))
    #这里的color_feature的维度仅有8个，结果不好，用颜色特征做效果并不好
    color_feature = color_moments(img)
    #这里imgPath.split(os.path.sep)[-1]是子目录下的文件名.jpg，而[-2]是子目录的名字,从而区分标签
    label = imgPath.split(os.path.sep)[-2]
    data.append(color_feature)
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
print("Using Time Is {}".format(endtime - starttime))

