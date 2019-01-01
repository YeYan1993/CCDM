import os
import pandas as pd
import numpy as np
import xml.etree.cElementTree as et
import nltk

# 读取files文件下下所有文件
xmlPath = "CCDM_data/Annotations"
files = os.listdir(xmlPath)

#创建空list 用来保存信息
filename_list = []
name_list = []
x_min_list = []
x_max_list = []
y_min_list = []
y_max_list = []

# 对files文件下所有xml文件进行遍历
for xml_file in files:
    a = xmlPath + '/' + xml_file #这里如果直接赋值xml_file，而不是a的话，就仅仅只是在最外层文件夹中寻找"IMG_000001.xml"文件，报错
    tree = et.parse(a) #创建一个xml树
    root = tree.getroot() #创建树的根目录
    filename = root.find('filename').text  #这里的filename值是取自根目录下的
    #在根目录下的object子目录下遍历需要的名字、位置信息
    Object = root.find('object')
    name = Object.find('name').text
    #bndbox在Oject 目录下 才有名字位置信息
    bndbox = Object.find('bndbox')
    x_min = bndbox.find('xmin').text
    x_max = bndbox.find('xmax').text
    y_min = bndbox.find('ymin').text
    y_max = bndbox.find('ymax').text

    #将所有信息保存
    name_list.append(name)
    filename_list.append(filename)
    x_min_list.append(x_min)
    x_max_list.append(x_max)
    y_min_list.append(y_min)
    y_max_list.append(y_max)

# 将所有的6个数组拼接为一个矩阵形式
# for list in [filename_list,name_list,x_min_list,x_max_list,y_min_list,y_max_list]:
#     list = np.array(list).reshape(-1,1)
filename_list = np.array(filename_list).reshape(-1,1)
name_list = np.array(name_list).reshape(-1,1)
x_min_list = np.array(x_min_list).reshape(-1,1)
x_max_list = np.array(x_max_list).reshape(-1,1)
y_min_list = np.array(y_min_list).reshape(-1,1)
y_max_list = np.array(y_max_list).reshape(-1,1)
matrix = np.hstack((filename_list,name_list,x_min_list,x_max_list,y_min_list,y_max_list))
print(matrix)

# 利用pandas模块将得到的信息保存为csv文件
bf_data = pd.DataFrame(matrix)
bf_data.columns = ['filename', 'category', 'x_min', 'x_max', 'y_min','y_max']
# bf_data.to_csv('bf_data.csv', encoding='utf-8') #这里必须将编码保存为utf-8的格式，不然再次打开，会报编码格式不对，保存的是GBK格
bf_data.to_csv('bv_data2.csv')

#读取所有的种类的个数
bf_category1 = nltk.FreqDist(bf_data['category'])
print(bf_category1)
bf_category2 = bf_data['category'].unique()
print(bf_category2)








