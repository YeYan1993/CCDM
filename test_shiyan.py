#
# a = ['啊123','啊23', '啊45' ,'哈1']
# b = ['啊', '哈']
# c = [10,11,12,13]
# for i in b:
#     for j in a:
#         if i in j:
#             aa = c[a.index(j)]
#
#
#
# s = [11, 22, 33, 44, 22, 11]
# from collections import defaultdict
# d = defaultdict(list)
# for k,va in [(v,i) for i,v in enumerate(s)]:
#     d[k].append(va)
# print (d)
#
# #取list重复值的索引
# l = ['a','b','c','c','d','c']
# find = 'c'
# for i,v in enumerate(l):
#     if v==find:
#         print(i)
#
# #图像数据格式转换
# from PIL import Image
#
# img = Image.open("butter_categories\\阿芬眼蝶\\AFae0022001xx01c.jpg")
# img.save("1.jpeg", format="jpeg")


# #图片生成器
# from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
#
# datagen = ImageDataGenerator(
#         rotation_range=40,
#         width_shift_range=0.2,
#         height_shift_range=0.2,
#         shear_range=0.2,
#         zoom_range=0.2,
#         horizontal_flip=True,
#         fill_mode='nearest')
#
# img = load_img("butter_categories\\阿芬眼蝶\\AFae0022001xx01c.jpeg")  # this is a PIL image
# x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
# x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
#
# # the .flow() command below generates batches of randomly transformed images
# # and saves the results to the `preview/` directory
# i = 0
# for batch in datagen.flow(x, batch_size=1,
#                           save_to_dir='preview', save_prefix='bf', save_format='jpeg'):
#     i += 1
#     if i > 20:
#         break

# # 将94个类取出来
# import os
# label_list = []
# for i in os.walk('butter_categories'):
#     if i[2] ==[]:
#         label_list = i[1]
#
# p = 0
# for category in label_list:
#     for q in os.walk('butter_categories'):
#         if q[1] == []:
#             if category in q[0]:
#                 p = p + 1
#                 print("{}类{}有{}个".format(p,category,len(q[2])))
#
#





#
# os.makedirs('random_pick_2')
# num = 0
# # for m in label_list:
# #     label_removed_list = label_list.remove(m)
# #     for n in label_removed_list:
# #         if
# #         num = num+1
# #         os.makedirs('random_{}'.format(num))
# for (m,n) in (label_list,label_list.remove(m)):


# import os
# import shutil
#
# #将butter_categories 文件夹下的子文件夹归类操作(聚类)
# lists = ['斑蝶','粉蝶','凤蝶','环蝶','灰蝶','喙蝶科','蛱蝶','绢蝶','弄蝶','蚬蝶科','眼蝶','珍蝶']
# for i in os.walk('butter_categories'):
#     if i[2] == []:
#         for list in lists:
#             for list_dir in i[1]:
#                 if list in list_dir:
#                     path_file = 'b_bf\\'+list
#                     shutil.move('butter_categories\\'+list_dir,path_file)
#                 # else:
#                 #     print('没有{}文件夹！'.format(list))
# # srcPath = "test1\\巴黎翠凤蝶"
# # destPath = "1"
# # shutil.copytree(srcPath,destPath)

# #图像输入算法实验（直接对大类中进行读取文件操作，然后对以大类作为类别标签）
# import cv2
# from imutils import paths
# import skimage.feature as ft
# import numpy as np
# import os
# #导入图像数据
# #imgPaths = list(paths.list_images("F:\pycharm_project\First_image_classifier\leedsbutterfly\images"))
# imgPaths = list(paths.list_images("all_sum_butter_categories"))
# data = []
# labels = []
#
# for (i, imgPath) in enumerate(imgPaths):
#     # img = cv2.imread(imgPath)
#     img = cv2.imdecode(np.fromfile(imgPath, dtype=np.uint8), -1)  # 这里由于有中文路径问题，无法用imread进行读取图片
#     img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#     #np.array.flatten()将数组变为一维
#     img = cv2.resize(img,(32,32))
#     feature = ft.local_binary_pattern(img, 12, 4, method='ror').flatten()
#     # feature, hog_image = ft.hog(img, orientations=9, pixels_per_cell=(6, 6),
#     #                             cells_per_block=(3, 3), transform_sqrt=None, visualise=True)
#     #这里imgPath.split(os.path.sep)[-1]是子目录下的文件名.jpg，而[-2]是子目录的名字,从而区分标签
#     label = imgPath.split(os.path.sep)[-3]
#     data.append(feature)
#     labels.append(label)
#     print("Image Loading ....")
#     print("Processed {}/{}".format(i+1,len(imgPaths)))

import numpy as np
a = [[2,3,4],
     [5,6,7]]


b = a[1,2]
print(b)