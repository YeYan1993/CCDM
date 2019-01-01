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
from sklearn.svm import SVC

#导入图像数据
#imgPaths = list(paths.list_images("F:\pycharm_project\First_image_classifier\leedsbutterfly\images"))
imgPaths = list(paths.list_images("leedsbutterfly\data_bf"))
data = []
labels = []

for (i, imgPath) in enumerate(imgPaths):
    img = cv2.imread(imgPath)
    #img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #np.array.flatten()将数组变为一维
    img = cv2.resize(img,(32,32),3).flatten()
    #这里imgPath.split(os.path.sep)[-1]是子目录下的文件名.jpg，而[-2]是子目录的名字,从而区分标签
    label = imgPath.split(os.path.sep)[-2]
    data.append(img)
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
#这里对分好之后的trainX进行交叉验证，这里交叉验证的目的是为了将模型中参数确定好，
# 还有就是可以相对客观的判断这些参数对训练集之外的数据的符合程度
#kf = KFold(trainX,n_folds=10)
#model = KNeighborsClassifier(p=1,n_neighbors=2,weights='uniform',algorithm='auto')
#score = cross_val_score(model,trainX,trainY,cv=10,scoring='accuracy')
#print("Accuracy is {}".format(score.mean()))

#分类器实验
#model = KNeighborsClassifier(p=1,n_neighbors=2,weights='uniform',algorithm='auto')
model = SVC(C=100, kernel = 'rbf', gamma = 'auto',class_weight = 'balanced')
#这里的c是错误项的惩罚系数。C越大，即对分错样本的惩罚程度越大，因此在训练样本中准确率越高，
# 但是泛化能力降低，也就是对测试数据的分类准确率降低。相反，减小C的话，容许训练样本中有一些误分类错误样本，泛化能力强。
model.fit(trainX,trainY)
#这里是将所有的训练数据集放到已经调优好的模型中去，再次训练，由于这里的测试集数据更大，fit效果更好
print(classification_report(testY,model.predict(testX)))