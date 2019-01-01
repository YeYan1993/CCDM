import numpy as np
import cv2
import os
from sklearn.model_selection import cross_val_score,train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import LabelEncoder,scale
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report,confusion_matrix
from skimage.feature import hog,local_binary_pattern
from sklearn.linear_model import LogisticRegression

from imutils import paths

import datetime
import pickle


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
    feature,hogimage= hog(img, orientations=9, pixels_per_cell=(6, 6),
                                cells_per_block=(3, 3), transform_sqrt=None, visualise=True)
    #这里imgPath.split(os.path.sep)[-1]是子目录下的文件名.jpg，而[-2]是子目录的名字,从而区分标签
    label = imgPath.split(os.path.sep)[-2]
    data.append(feature)
    labels.append(label)
    print("Image Loading ....")
    print("Processed {}/{}".format(i+1,len(imgPaths)))


data1 = np.array(data)
labels1 = np.array(labels)

#规则化
data2 = scale(data1)
#将labels1 转成整形表示
le = LabelEncoder()
labels2 = le.fit_transform(labels1)



#将数据集进行分类
(trainX,testX,trainY,testY) = train_test_split(data2,labels2,test_size=0.25,random_state=42)
#model = SVC(C=10, kernel = 'rbf', gamma = 'auto',class_weight = 'balanced')

# n_estimators is the number of trees we want to make
# min_samples_split is the minimum number of rows we need to make a split
# min_samples_leaf is the minimum number of samples we can have at the place where a tree branch ends (the bottom points of the tree)
# model = RandomForestClassifier(random_state=1, n_estimators=200, min_samples_split=4, min_samples_leaf=2)
# model = AdaBoostClassifier()
model = LogisticRegression(random_state=1)

score = cross_val_score(model,trainX,trainY,cv = 10,scoring='accuracy')
print(score.mean())
#分类器实验
#model = SVC(C=10, kernel = 'rbf', gamma = 'auto',class_weight = 'balanced')
model.fit(trainX,trainY)
#保存模型
#pickle.dump(model,open('model_save/model.model','wb'))
#导入模型
# model = pickle.load(open('model_save/model.model','wb'))
print(classification_report(testY,model.predict(testX)))

#看混淆矩阵的pt值
pt = confusion_matrix(testY,model.predict(testX))
# print(pt)


#随机挑选1个样本进行测试
instance = testX[100:109]
#print(instance)
a = instance[0] #这里的a值是列的形式，而model.predict的单个输入必须是行的形式，因此要reshape
test1 = model.predict(a.reshape(1,-1))
instance_label = testY[100:109]
true1 = instance_label[0]
print('instance[0] prediction class is {}'.format(test1))
print('instance[0] true class is {}'.format(true1))

endtime = datetime.datetime.now()
print("Using Time Is {}".format(endtime - startime))