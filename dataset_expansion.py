import numpy as np
from PIL import Image,ImageEnhance
import cv2
import matplotlib.pyplot as plt
import os
from keras.preprocessing.image import ImageDataGenerator,array_to_img,img_to_array,load_img

# def randomRotation(image):
#     """
#           对图像进行随机任意角度(0~360度)旋转
#          :param mode 邻近插值,双线性插值,双三次B样条插值(default)
#          :param image PIL的图像image
#          :return: 旋转转之后的图像
#     """
#     random_angle = np.random.randint(-15, 15)
#     return image.rotate(random_angle)
#
#
# def randomColor(image):
#     """
#         对图像进行颜色抖动
#         :param image: PIL的图像image
#         :return: 有颜色色差的图像image
#     """
#     random_factor = np.random.randint(8, 12) / 10.
#
#   # 随机因子
#     color_image = ImageEnhance.Color(image).enhance(random_factor)
#  # 调整图像的饱和度
#     random_factor = np.random.randint(8, 12) / 10.
#  # 随机因子
#     brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)
#   # 调整图像的亮度
#     random_factor = np.random.randint(8, 12) / 10.
#   # 随机因1子
#     contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)
#   # 调整图像对比度
#     random_factor = np.random.randint(8, 10) / 10.
#    # 随机因子
#     return ImageEnhance.Sharpness(contrast_image).enhance(random_factor)
#      # 调整图像锐度
#
# image = Image.open("butter_categories\\阿芬眼蝶\\AFae0022001xx01c.jpeg")
# image = randomColor(image)
# plt.figure("bf")
# plt.imshow(image)
# plt.show()



#定义一个图像数据扩容的函数dataset_expansion(输入路径，输出路径，输出图片的个数)
def dataset_expansion(input_path,output_path,k):
    datagen = ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')

    img = load_img(input_path)  # this is a PIL image
    x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
    x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

    # the .flow() command below generates batches of randomly transformed images
    # and saves the results to the `preview/` directory
    i = 0
    for batch in datagen.flow(x, batch_size=1,save_to_dir=output_path, save_prefix='bf_expansion', save_format='jpeg'):
        i += 1
        if i > k:
            break


#定义一个函数，查找文件夹下所有的.jpg文件，对每一个文件进行dataset_expansion操作

#1.获取大文件下所有的子文件名称（标签），放到一个label_list里面去
label_list = []
for i in os.walk('leedsbutterfly/data_bf'):
    if i[2] ==[]:
        label_list = i[1]
    #2.对所有文件进行dataset_expansion操作
    elif i[2] != []:
        m = 0
        for i_index in i[2]:
            dataset_expansion(os.path.join(i[0],i_index),i[0],20)
            m = m+1
            print("第{}正在数据扩容中......".format(m))