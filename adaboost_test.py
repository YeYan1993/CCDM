import numpy as np
import cv2
import os
from sklearn.model_selection import cross_val_score,train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import LabelEncoder,scale
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from skimage.feature import local_binary_pattern
from imutils import paths
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
data2 = scale(data1)
#将labels1 转成整形表示
le = LabelEncoder()
labels2 = le.fit_transform(labels1)



#基于权重w的wknn分类器
def wk_knn(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize,1)) - dataSet

    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()
    classCount={}
    w=[]
    for i in range(k):
        w.append((distances[sortedDistIndicies[k-1]]-distances[sortedDistIndicies[i]]\
        )/(distances[sortedDistIndicies[k-1]]-distances[sortedDistIndicies[0]]))
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + w[i]
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


#将数据集进行分类
(trainX,testX,trainY,testY) = train_test_split(data2,labels2,test_size=0.25,random_state=42)
#model = SVC(C=10, kernel = 'rbf', gamma = 'auto',class_weight = 'balanced')
#score = cross_val_score(model,trainX,trainY,cv = 10,scoring='accuracy')
#print(score.mean())
#分类器实验
# model = SVC(C=10, kernel = 'rbf', gamma = 'auto',class_weight = 'balanced')
#model = KNeighborsClassifier(n_neighbors=2,n_jobs=1)
#model.fit(trainX,trainY)
#print(classification_report(testY,model.predict(testX)))
# ada = AdaBoostClassifier(base_estimator=model,n_estimators=2)

res = wk_knn(trainX,trainY,5)

score = cross_val_score(ada,trainX,trainY,cv = 10,scoring='accuracy')
print(score.mean())



endtime = datetime.datetime.now()
print("Using Time Is {}".format(endtime - startime))