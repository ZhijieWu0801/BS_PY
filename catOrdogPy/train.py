epochs = 1
modelName = "my_model"+"_"+str(epochs)+".h5"
trainImgPath = r"./data/train/"
testImgPath = r"./data/test/"
savePath = "./model/"+modelName
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import os, random
import matplotlib.pyplot as plt
from keras.models import load_model
from matplotlib.pyplot import imshow
import numpy as np
from PIL import Image
import pylab

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def define_cnn_model():
    # 使用Sequential序列模型
    model = Sequential()
    # 卷积层
    model.add(Conv2D(32, (3, 3), activation="relu", padding="same",
                     input_shape=(200, 200, 3)))  # 第一层即为卷积层，要设置输入进来图片的样式  3是颜色通道个数
    # 最大池化层
    model.add(MaxPool2D((2, 2)))  # 池化窗格
    # Flatten层
    model.add(Flatten())
    # 全连接层
    model.add(Dense(128, activation="relu"))  # 128为神经元的个数
    model.add(Dense(1, activation="sigmoid"))
    # 编译模型
    opt = SGD(lr=0.001, momentum=0.9)  # 随机梯度
    model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])
    return model


def train_cnn_model():
    # 实例化模型
    model = define_cnn_model()
    # 创建图片生成器
    datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    train_it = datagen.flow_from_directory(
        trainImgPath,
        class_mode="binary",
        batch_size=64,
        target_size=(200, 200))  # batch_size:一次拿出多少张照片 targe_size:将图片缩放到一定比例
    # 训练模型
    model.fit_generator(train_it,
                        steps_per_epoch=len(train_it),
                        epochs=epochs,
                        verbose=1)
    model.save(savePath)


train_cnn_model()

# -*- coding: UTF-8 -*-
"""
=====验证=====
"""
usePath = savePath
# usePath = "./model/my_model_1.h5"
model_path = usePath
model = load_model(model_path)

plt.rcParams['font.sans-serif'] = ['SimHei']


#
# 在文件夹拿随机图片
#
def read_random_image():
    folder = testImgPath
    file_path = folder + random.choice(os.listdir(folder))
    pil_im = Image.open(file_path, 'r')
    return pil_im


def get_predict(pil_im, model):
    # 首先更改图片的大小
    name = ''
    pil_im = pil_im.resize((200, 200))
    # 将格式转为numpy array格式
    array_im = np.asarray(pil_im)
    # array_im = array_im.resize((4,4))
    array_im = array_im[np.newaxis, :]
    # 对图像检测
    result = model.predict([[array_im]])
    if result[0][0] > 0.5:
        name = "它是狗！"
        print("预测结果是：狗")
    else:
        name = "它是猫！"
        print("预测结果是：猫")
    return name


pil_im = read_random_image()
imshow(np.asarray(pil_im))
plt.title(get_predict(pil_im, model))
pylab.show()
