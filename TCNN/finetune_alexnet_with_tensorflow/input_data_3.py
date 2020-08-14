import h5py
import os
import numpy as np
import cv2
import glob
import h5py
import keras
from sklearn.model_selection import train_test_split

Normal_dir = './Normal/'
Glaucoam_dir = './Glaucoma/'
INPUT_DATA = './RIM-ONE2/'
test_DATA = './RIM-ONE2_test/'


def create_image_lists():
    # f = h5py.File("dataset0809_l_r_old.hdf5", "w")
    f = open('train.txt', 'w')
    val = open('val.txt', 'w')
    name = []
    label = []
    sub_dirs = [x[0] for x in os.walk(INPUT_DATA)]  # 获取所有子目录
    is_root_dir = True  # 第一个目录为当前目录，需要忽略
    # 分别对每个子目录进行操作
    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False
            continue
        # 获取当前目录下的所有有效图片
        extensions = {'bmp', 'jpg'}
        file_list = []  # 存储所有图像
        dir_name = os.path.basename(sub_dir)  # 获取路径的最后一个目录名字
        for extension in extensions:
            file_glob = os.path.join(INPUT_DATA, dir_name, '*.' + extension)
            file_list.extend(glob.glob(file_glob))
        if not file_list:
            continue
        label_name = dir_name
        for file_name in file_list:
            name.append(file_name)
            if label_name == 'Glaucoma':
                label.append('1')
            else:
                label.append('0')
    train = np.asarray(name)
    label = np.asarray(label)
    train_name, test_name, train_l, test_l = train_test_split(train, label, train_size=0.25, random_state=20)
    for (x, y) in zip(train_name, train_l):
        s = x + ' ' + y + '\n'
        f.writelines(s)
    f.close()
    for (x, y) in zip(test_name, test_l):
        s = x + ' ' + y + '\n'
        val.writelines(s)
    val.close()



def load_data():
    f = h5py.File('dataset0809_l_r_old.hdf5', 'r')
    imgs = f['imgs'][:]
    labels = f['labels'][:]
    f.close()
    print(imgs[0].shape)
    return imgs, labels  # training_imgs, test_imgs, training_labels, test_labels


if __name__ == '__main__':
    # load_data()
    create_image_lists()
