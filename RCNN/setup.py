import os
import shutil
import config


def mkdir(path):
    if os.path.exists(path):
        # 递归删除文件夹下的所有子文件夹和子文件
        shutil.rmtree(path)
    # 创建目录
    os.mkdir(path)


if __name__ == '__main__':
    # save fine-tune data
    mkdir(config.FINE_TUNE_DATA)
    # save pre-train model
    mkdir(config.SAVE_MODEL_PATH.strip().rsplit('/', 1)[0])
    # save fine-tune model
    mkdir(config.FINE_TUNE_MODEL_PATH.strip().rsplit('/', 1)[0])
    # save svm model and data
    mkdir(os.path.join(config.TRAIN_SVM, '1'))
    mkdir(os.path.join(config.TRAIN_SVM, '2'))