import numpy as np
import pandas as pd
import os
import xlrd
import re
import shutil
from collections import defaultdict
from PIL import Image


workbook = xlrd.open_workbook("模式照蝴蝶命名20180226.xlsx")
#打印每个sheet的名字
print(workbook.sheet_names())
booksheet = workbook.sheet_by_index(0)
# booksheet = workbook.sheet_by_name("命名编号")

#读一列模式照数据 都是list形式
categories_col = booksheet.col_values(7)
name_col = booksheet.col_values(11)

#读取蝴蝶生态照中需要的94个蝴蝶种类名称
bf_data = pd.read_csv("bf_data.csv")
ecological_bf = bf_data["category"].unique()

#2次循环：对取出上面categories_col列进行循环遍历，如果与上面名字相同，获取其索引

index = []
#这里如果直接拿categries的索引去取name_col的值，只会取第一个
#这里涉及到list中取重复值的索引 具体见aa.py文件
for eco_name in ecological_bf:
    for i,name in enumerate(categories_col):
        if (eco_name in name):
            file_nm = name_col[i]
            index.append(file_nm+'.jpg')



# os.makedirs("butter_moshi")
print(os.listdir("蝴蝶图库"))

#自己写的递归函数
# def search_file(path, img_name):
#     #判断路径是否存在
#     if(os.path.exists(path)):
#         #获得该目录下所有文件或者所有目录files
#         files = os.listdir(path)
#         #遍历
#
#         for file in files:
#             #将路径和文件进行拼接
#             m = os.path.join(path,file)
#             #判断该路径是否是文件夹
#             a = os.path.isfile(m)
#             b = re.findall(img_name, m)
#             if (os.path.isfile(m)) and (len(re.findall(img_name, m)) > 0):
#                 shutil.copy(m, "butter_moshi")
#             else:
#                 search_file(m, img_name)

#利用os.walk返回的是一个迭代器 ,由于是要进行复制操作，操作一次就可以了
# #这里是对蝴蝶图库这个总的目录下进行提取目标jpg图像，这里的参数img_name是名字没有以.jpg结尾
# def search_file(path,img_name,file_name):
#     for i in os.walk(path):
#         if i[2] != []:
#             for i_index in i[2]:
#                 #这里因为在蝴蝶图库中还有png图像和jpg图像
#                 if img_name in i_index:
#                     # if 完全相等，直接进行复制
#                     if 'jpeg' or  'jpg' in i_index:
#                         shutil.copy(i[0]+"\\"+i_index,file_name)
#                     #elif 图片中有png格式的照片，进行转换为jpeg格式
#                     elif 'png' in i_index:
#                         img = Image.open(i[0]+"\\"+i_index)
#                         img.save(file_name+"\\"+img_name+".jpeg",format('jpeg'))


#这里是对生态照片butter_shengtai文件下进行遍历，由于img_name参数的值是以.jpg为后缀的，所以需要重新写
def search_file(path,img_name,file_name):
    for i in os.walk(path):
        if i[2] != []:
            for i_index in i[2]:
                if img_name == i_index:
                        shutil.copy(i[0]+"\\"+i_index,file_name)

#现在取出了94个类的所有文件，进行分类操作
print(ecological_bf)

def img_classification(list1,list2,path1):
    for categary in ecological_bf:
        #1.创建文件夹,已经创建好了
        path2 = 'butter_categories\\'+categary
        # os.makedirs(path2) #这里由于第一次已经建好文件加了，就需要注释掉

        #2.取值
        ind = []
        for i,v in enumerate(list1):
            if categary in v:
                ind.append(list2[i])

        #3.将jpg、png文件复制到创建好的文件夹下
        for index_name in ind:
            search_file(path1, index_name , path2)
        print("{}已完成复制！".format(categary))
        print("该种类有{}个".format(len(ind)))


# img_classification(categories_col,name_col,'蝴蝶图库')
img_classification(bf_data["category"],bf_data["filename"],'butter_shengtai')