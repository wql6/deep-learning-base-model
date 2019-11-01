# -*- coding: utf-8 -*-
"""
__author__ = 'Alex wu'
__version__ = '1.0'
"""

# 导入模块
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
# 读取MNIST数据
mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
#    重构数据至4维（样本，像素X，像素Y，通道）

x_train=x_train.reshape(x_train.shape+(1,))
x_test=x_test.reshape(x_test.shape+(1,))
# print(x_train[0, :])
# 归一化
x_train, x_test = x_train/255.0, x_test/255.0
# print(x_train[0, :])
#     数据标签
label_train = keras.utils.to_categorical(y_train, 10)
label_test = keras.utils.to_categorical(y_test, 10)

print(x_train.shape)
print(x_test.shape)
# print(x_train.shape)
print(label_train.shape)
print(type(y_test))
print(y_test)
print(len(y_test))
print(y_test.shape)
# LeNet-5构筑
model = keras.Sequential([
    keras.layers.Conv2D(6, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='valid'),
    keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'),
    keras.layers.Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='valid'),
    keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'),
    keras.layers.Flatten(),
    keras.layers.Dense(120, activation='sigmoid'),
    keras.layers.Dense(84, activation='sigmoid'),
    keras.layers.Dense(10, activation='softmax')])
# 使用SGD编译模型
model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=['accuracy'])  # SGD 是单个运算，精度高，不过也容易过拟合，速度慢
# 学习30个纪元（可依据CPU计算力调整），使用20%数据交叉验证
records = model.fit(x_train, label_train, epochs=1, validation_split=0.2, )


# model.metrics_names=['loss', 'acc', 'mean_pred']
# model = keras.models.load_model('my_modle.h5')
# score = model.evaluate(x_test, label_test, batch_size=100, verbose=1, sample_weight=None)
# print(model.metrics_names)
# print(score)
# print(type(score))


modle.save('my_model.h5')
# # 预测
print('predict==========')
print(model.predict(x_test))
y_pred = np.argmax(model.predict(x_test), axis=1)
print('y_pred:',y_pred)
print("prediction accuracy: {}".format(sum(y_pred==y_test)/len(y_test)))
model.summary()


# 绘制结果
# plt.plot(records.history['loss'],label='training set loss')
# plt.plot(records.history['val_loss'],label='validation set loss')
# plt.ylabel('categorical cross-entropy'); plt.xlabel('epoch')
# plt.legend()