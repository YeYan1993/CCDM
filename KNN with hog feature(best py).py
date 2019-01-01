import cv2
import numpy as np
from imutils import paths
import os
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder,scale
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix #查看预测结果
from skimage import feature as ft
from sklearn.cross_validation import cross_val_score
from sklearn.feature_selection import SelectKBest,chi2 #卡方检验
from sklearn.decomposition import PCA
import pickle #导入和生成模型
import datetime

startime = datetime.datetime.now()

#导入图像数据
#imgPaths = list(paths.list_images("F:\pycharm_project\First_image_classifier\leedsbutterfly\images"))
imgPaths = list(paths.list_images("leedsbutterfly/data_bf"))
# imgPaths = list(paths.list_images("train_4000"))
data = []
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
    data.append(feature)
    labels.append(label)
    print("Image Loading ....")
    print("Processed {}/{}".format(i+1,len(imgPaths)))


data1 = np.array(data)
labels1 = np.array(labels)


#将labels1 转成整形表示
le = LabelEncoder()
labels2 = le.fit_transform(labels1)
#用卡方检验选择 与目标相关性高的特征 KNN 的k=380 ，精度在0.64，SVM的k=500 ，精度在0.63
data2 = SelectKBest(chi2, k=380).fit_transform(data1, labels2) #这里的data1必须是非负矩阵，所以需要先用卡方检验在标准化
#对特征选择之后想通过PCA无监督降维的方式提高效率，但是发现精度也相应减少，由原本的0.63减少到0.58
# data2 = PCA(n_components=100).fit_transform(data2, labels2)
#将每一列特征进行标准化正太分布（均值为0，方差为1）
data2 = scale(data2) #这里的data2是含负数的矩阵


#将数据集进行分类
(trainX,testX,trainY,testY) = train_test_split(data2,labels2,test_size=0.25,random_state=42)
model = SVC(C=10, kernel = 'rbf', gamma = 'auto',class_weight = 'balanced') #SVM中卡方检验用的K = 500
# model =KNeighborsClassifier(p=1, n_neighbors=92, weights='uniform', algorithm='auto')
score = cross_val_score(model,trainX,trainY,cv = 10,scoring='accuracy')
print(score.mean())
# model =KNeighborsClassifier(p=1, n_neighbors=92, weights='uniform', algorithm='auto')
# model = SVC(C=10, kernel = 'rbf', gamma = 'auto',class_weight = 'balanced',probability=True)
model.fit(trainX,trainY)
#保存模型
# pickle.dump(model,open('model_save/model.model','wb'))
#导入模型
# model = pickle.load(open('model_save/model.model','wb'))
print(classification_report(testY,model.predict(testX)))

#看混淆矩阵的pt值
pt = confusion_matrix(testY,model.predict(testX))
print(pt)


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
