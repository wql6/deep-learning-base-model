from __future__ import division, print_function, absolute_import
import pickle
import numpy as np 
import config
import os.path
import codecs
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
import preprocessing_RCNN as prep
import cv2


def load_data(datafile, num_class, save=False, save_path='dataset.pkl'):
    fr = codecs.open(datafile, 'r', 'utf-8')
    train_list = fr.readlines()
    labels = []
    images = []
    for line in train_list:
        tmp = line.strip().split(' ')
        fpath = tmp[0]
        img = cv2.imread(fpath)
        # 样本resize到227x227，转为矩阵保存
        img = prep.resize_image(img, config.IMAGE_SIZE, config.IMAGE_SIZE)
        np_img = np.asarray(img, dtype="float32")
        images.append(np_img)

        index = int(tmp[1])
        label = np.zeros(num_class)
        label[index] = 1
        labels.append(label)
    if save:
        pickle.dump((images, labels), open(save_path, 'wb'))
    fr.close()
    return images, labels

# pickle读取，反序列化
def load_from_pkl(dataset_file):
    X, Y = pickle.load(open(dataset_file, 'rb'))
    return X,Y


# Building 'AlexNet'
def create_alexnet(num_classes):
    network = input_data(shape=[None, config.IMAGE_SIZE, config.IMAGE_SIZE, 3])
    # 4维输入张量，卷积核个数，卷积核尺寸，步长
    network = conv_2d(network, 96, 11, strides=4, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    # 数据归一化
    network = local_response_normalization(network)
    network = conv_2d(network, 256, 5, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = conv_2d(network, 384, 3, activation='relu')
    network = conv_2d(network, 384, 3, activation='relu')
    network = conv_2d(network, 256, 3, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = fully_connected(network, 4096, activation='tanh')
    network = dropout(network, 0.5)
    network = fully_connected(network, 4096, activation='tanh')
    network = dropout(network, 0.5)
    network = fully_connected(network, num_classes, activation='softmax')
    momentum = tflearn.Momentum(learning_rate=0.001, lr_decay=0.95, decay_step=200)
    network = regression(network, optimizer=momentum,
                         loss='categorical_crossentropy')
    return network


def train(network, X, Y, save_model_path):
    # Training
    model = tflearn.DNN(network, checkpoint_path='model_alexnet',
                        max_checkpoints=1, tensorboard_verbose=2, tensorboard_dir='output')
    if os.path.isfile(save_model_path + '.index'):
        model.load(save_model_path)
        print('load model...')

    model.fit(X, Y, n_epoch=2, validation_set=0.1, shuffle=True,
              show_metric=True, batch_size=64, snapshot_step=200,
              snapshot_epoch=False, run_id='alexnet_oxflowers17') # epoch = 1000
    # Save the model
    model.save(save_model_path)
    print('save model...')


def predict(network, modelfile, images):
    model = tflearn.DNN(network)
    model.load(modelfile)
    return model.predict(images)


if __name__ == '__main__':
    # 大数据及(17flowers)上训练alexnet
    X, Y = load_data(config.TRAIN_LIST, config.TRAIN_CLASS)
    net = create_alexnet(config.TRAIN_CLASS)
    train(net, X, Y, config.SAVE_MODEL_PATH)

