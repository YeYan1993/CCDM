# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 14:11:28 2018

@author: yuyangyang
"""

from keras.models import model_from_json
from skimage import io,transform
import keras
import h5py
import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片
lena = mpimg.imread('E:\\image project\\dogvscat\\dog_vs_cat\\data\\test1\\6618.jpg') # 读取和代码处于同一目录下的
plt.imshow(lena)
plt.show()  # 显示图片
#lena = lena.reshape(208, 208,3).astype('float32')
img=transform.resize(lena,(208,208,3)) #已经做了一次归一化
lena = img.reshape(1,208, 208,3).astype('float32')
#lena = lena/255    #统一格式
model = model_from_json(open('E:\\2018.3\\Image_classification\\0319_model_architecture.json').read())
model.load_weights('E:\\2018.3\\Image_classification\\0319_model_weights.h5')    #加载模型
pre=model.predict_classes(lena)   #  预测model.predict_classes
print(pre)