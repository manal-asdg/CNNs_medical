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
    f = h5py.File("RIM-ONE2_vgg16_good_RGB.hdf5", "w")
    training_imgs = []
    training_labels = []
    sub_dirs = [x[0] for x in os.walk(INPUT_DATA)]  # subdirs
    is_root_dir = True  # ignore first dir
    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False
            continue
        # get the pics by extensions
        extensions = {'bmp', 'jpg'}
        file_list = []  # store file
        dir_name = os.path.basename(sub_dir)  # get label name
        for extension in extensions:
            file_glob = os.path.join(INPUT_DATA, dir_name, '*.' + extension)
            file_list.extend(glob.glob(file_glob))
        if not file_list:
            continue
        label_name = dir_name
        for file_name in file_list:
            tmp_img = cv2.imread(file_name)
            tmp_img = tmp_img[:, :, :: -1] # change to rgb
            tmp_img = cv2.resize(tmp_img, (224, 224))
            training_imgs.append(tmp_img)
            if label_name == 'Glaucoma':
                training_labels.append(1)
            else:
                training_labels.append(0)
    training_imgs = np.asarray(training_imgs)
    training_labels = np.asarray(training_labels)
    print(training_imgs.shape, training_labels.shape)
    training_labels = keras.utils.to_categorical(training_labels, num_classes=2)
    f.create_dataset('imgs', data=training_imgs)
    f.create_dataset('labels', data=training_labels)
    for key in f.keys():
        print(key)
    f.close()


def load_data():
    f = h5py.File('RIM-ONE2_vgg16_good_RGB.hdf5', 'r')
    imgs = f['imgs'][:]
    labels = f['labels'][:]
    f.close()
    return imgs, labels


def extract_data():
    f = h5py.File('RIM-ONE2_vgg16_good.hdf5', 'r')
    imgs = f['imgs'][:]
    labels = f['labels'][:]
    f.close()
    i = 0
    for (x, y) in zip(imgs, labels):
        i = i + 1
        if y[0] == 1:
            cv2.imwrite('./extract/Galucoma/' + str(i) + '.jpg', x)
        else:
            cv2.imwrite('./extract/Normal/' + str(i) + '.jpg', x)


if __name__ == '__main__':
    # load_data()
    create_image_lists()
    # extract_data()
