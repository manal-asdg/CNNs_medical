from sklearn.model_selection import KFold
import input_data_2
import numpy
import keras
import h5py

# X, Y = input_data_2.load_data()
# i = 1
# kfold = KFold(n_splits=5, shuffle=True, random_state=7)
# f = h5py.File("kfold2.hdf5", "w")
# for train, test in kfold.split(X, Y):
#     f.create_group("fold{}".format(i))
#     f.create_dataset("fold{}/train_imgs".format(i), data=X[train])
#     f.create_dataset("fold{}/train_labels".format(i), data=Y[train])
#     f.create_dataset("fold{}/test_imgs".format(i), data=X[test])
#     f.create_dataset("fold{}/test_labels".format(i), data=Y[test])
#     i = i + 1
#
# for key in f.keys():
#     print(key)
# f.close()





def load_data(i):
    f = h5py.File("kfold2.hdf5", "r")
    train_imgs = f["fold{}/train_imgs".format(i)][:]
    train_labels = f["fold{}/train_labels".format(i)][:]
    test_imgs = f["fold{}/test_imgs".format(i)][:]
    test_labels = f["fold{}/test_labels".format(i)][:]
    f.close()
    # print(training_imgs.shape, training_labels.shape)
    # print(test_imgs.shape, test_labels.shape)
    print(train_imgs[0].shape)
    return train_imgs, train_labels, test_imgs, test_labels


# train_imgs, train_labels, test_imgs, test_labels = load_data(2)