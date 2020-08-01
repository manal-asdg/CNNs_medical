# coding=utf-8
from __future__ import print_function
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.models import load_model
from keras.models import model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
import tensorflow as tf
import pickle
# import json
import time
import my_parser as ps
import numpy as np
from sklearn.model_selection import train_test_split
# import sys
import input_data_2
import keras
from matplotlib import pyplot as plt
import h5py
from sklearn.metrics import roc_curve
from sklearn.metrics import auc


nb_classes = 2
batch_size = 4
batch_size2 = 64
nb_epoch = 12
nb_epoch2 = 12
SEMI_TIMES = 20
validPercent = 10

LOAD_FLAG = True



LOAD_MODEL_FILE = "kf_2.hdf5"
MODEL_FILE = "se-kf2-2.h5"
# MODEL_FILE = sys.argv[2]
# dataPath = sys.argv[1]a

# slow data
# labels[0-9][0-499][0-3071]

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allocator_type = 'BFC'
config.gpu_options.per_process_gpu_memory_fraction = 0.80


#ts = time.time()
print("Loading labeled data......")
#with open('train_data', 'rb') as fo:
#    labels = pickle.load(fo, encoding='bytes')
# labels = pickle.load(open(dataPath + 'all_label.p', "rb"))
#X_train, Y_train = ps.parseTrain(labels, nb_classes, 'rgb')
# print('labels[9][499]', labels[9][499][1023], labels[9][499][2047], labels[9][499][3071])
# print('X_train[4999][31][31]', X_train[4999][31][31])
# X_train, Y_train = input_data_2.load_data()
# print("Y_train:", Y_train)
# Y_train = keras.utils.to_categorical(Y_train, num_classes=2)
# print("reshape Y_train:", Y_train)
# print("Y:",Y_train)
#te = time.time()
#print(te-ts, 'secs')
#
#ts = time.time()
#print("Loading test data......")
#with open('test_data', 'rb') as fo:
#    tests = pickle.load(fo, encoding='bytes')
# labels = pickle.load(open(dataPath + 'all_label.p', "rb"))
#X_test, Y_test = ps.parseTest(tests, nb_classes, 'rgb')
# print('labels[9][499]', labels[9][499][1023], labels[9][499][2047], labels[9][499][3071])
# print('X_train[4999][31][31]', X_train[4999][31][31])
# X_train, Y_train = input_data_2.load_data()
# print("Y_train:", Y_train)
# Y_train = keras.utils.to_categorical(Y_train, num_classes=2)
# print("reshape Y_train:", Y_train)
# print("Y:",Y_train)
#te = time.time()
#print(te-ts, 'secs')

f = h5py.File("kfold2.hdf5","r")
X_train = f["fold2/train_imgs"][:]
Y_train = f["fold2/train_labels"][:]
X_test = f["fold2/test_imgs"][:]
Y_test = f["fold2/test_labels"][:]
f.close()

print('shape: X_train', X_train.shape, 'Y_train', Y_train.shape, 'X_test', X_test.shape, 'Y_test', Y_test.shape)
 

# X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.2, random_state=1, shuffle=True)

# unlabels[0-44999][0-3071]
#ts = time.time()
print("Loading unlabeled data......")
f = h5py.File("unlabel_data2.h5","r")
X_unlabel = f["data"][:]
Y_unlabel = f["labels"][:]
# unlabels = pickle.load(open(dataPath + 'all_unlabel.p', "rb"))
#X_unlabel, Y_unlabel = ps.parseUnlabel(unlabels, nb_classes, 'rgb')
# print('unlabels[44999]', unlabels[44999][1023], unlabels[44999][2047], unlabels[44999][3071])
# print('X_unlabel[44999][31][31]', X_unlabel[44999][31][31])
#te = time.time()
#print(te-ts, 'secs')

# tests['ID'][0-9999], tests['data'][0-9999], tests['labels'][0-9999]
# ts = time.time()
# print("Loading test data......")
# with open('test_batch', 'rb') as fo:
#     tests = pickle.load(fo, encoding='bytes')
# # tests = pickle.load(open(dataPath + 'test.p', "rb"))
# X_test, Y_test = ps.parseTest(tests, nb_classes, 'rgb')
# # print('tests["data"][0]', tests["data"][0][0], tests["data"][0][1024], tests["data"][0][2048])
# # print('X_test[0][0][0]', X_test[0][0][0])
# te = time.time()
# print(te-ts, 'secs')

# pickle.dump((X_train, Y_train), open("fast_all_label", "wb"), True)
# pickle.dump((X_unlabel, Y_unlabel), open("fast_all_unlabel", "wb"), True)
# pickle.dump((X_test, Y_test), open("fast_test", "wb"), True)

# fast data
# ts = time.time()
# (X_train, Y_train) = pickle.load(open("fast_all_label", "rb"))
# (X_unlabel, Y_unlabel) = pickle.load(open("fast_all_unlabel", "rb"))
# (X_test, Y_test) = pickle.load(open("fast_test", "rb"))
# te = time.time()

X_train = X_train.astype('float32') / 255
X_unlabel = X_unlabel.astype('float32') / 255
X_test = X_test.astype('float32') / 255

'''(5000, 32, 32, 3) (5000, 10) (10000, 32, 32, 3) (10000, 10) (45000, 32, 32, 3) (45000, 10)'''
print('shape: X_train', X_train.shape, 'Y_train', Y_train.shape, 'X_test', X_test.shape, 'Y_test', Y_test.shape, 'X_unlabel', X_unlabel.shape, 'Y_unlabel', Y_unlabel.shape)

# add model
model = Sequential()
if LOAD_FLAG:
    model = load_model(LOAD_MODEL_FILE)
    # weight = np.load("vgg1618.0.npy")
    # model.set_weights(weight)
else:
    model.add(Convolution2D(32, 4, 4, border_mode='same', input_shape=X_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 4, 4))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.55))

    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.55))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.55))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    # start CNN
    sgd = SGD(lr=0.005, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    X_validation, Y_validation = ps.parseValidation(X_train, Y_train, nb_classes, len(X_train)*validPercent/100, _type='rgb')

    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, validation_data=(X_validation, Y_validation), shuffle=True)

# semi-supervised
# X_train_semi_prime = np.concatenate((X_unlabel, X_test), axis=0)
# Y_train_semi_prime = np.concatenate((Y_unlabel, Y_test), axis=0)
X_train_semi_prime = X_unlabel
Y_train_semi_prime = Y_unlabel
earlystopping = EarlyStopping(monitor='val_loss', patience=15, verbose=1) # 当监测值不再改善时，该回调函数将中止训练

for i in range(SEMI_TIMES):
    
    nb_epoch2 = nb_epoch2+4*i
    
    print("semi iter", i)

    print(" === Predicting unlabeled data...... === \n")
    Y_train_semi_prime = model.predict(X_train_semi_prime)
    # Y_unlabel = model.predict(X_unlabel)
    print(Y_train_semi_prime)
    # print(Y_unlabel)
    print(" === Parsing unlabeled data...... === \n")
#    if i == 0:
#        X_train_semi, Y_train_semi, index = ps.parseSemi(X_train_semi_prime, Y_train_semi_prime, 0.98) # 返回最大标签概率超过0.8的
#    elif i == 1:
#        X_train_semi, Y_train_semi, index = ps.parseSemi(X_train_semi_prime, Y_train_semi_prime, 0.98)
#    else:
#        X_train_semi, Y_train_semi, index = ps.parseSemi(X_train_semi_prime, Y_train_semi_prime, 0.98)
    
    nor_data = []
    gla_data = []
    index_nor = []
    index_gla = []
    
    for j in range(Y_train_semi_prime.shape[0]):
        if Y_train_semi_prime[j][0]>Y_train_semi_prime[j][1]:
            nor_data.append(Y_train_semi_prime[j][0])
            index_nor.append(j)
        else:
            gla_data.append(Y_train_semi_prime[j][1])
            index_gla.append(j)
    
    normal = {}
    glaucoma = {}
    
    for k in range(len(nor_data)):
        normal[index_nor[k]] = [nor_data[k]]
    for l in range(len(gla_data)):
        glaucoma[index_gla[l]] = [gla_data[l]]
    
    a = sorted(normal.items(),key=lambda item:item[1],reverse=True)
    b = sorted(glaucoma.items(),key=lambda item:item[1],reverse=True)
    
    nor_index = []
    gla_index = []
    
    nor_num = 30+3*i
    glau_num = 2*nor_num
    
    if len(nor_data)>nor_num and len(gla_data)>glau_num:
        for m in range(nor_num):
            nor_index.append(a[m][0])
        for n in range(glau_num):
            gla_index.append(b[n][0])
    else:
        break
    
    index = np.concatenate((nor_index,gla_index),axis=0)
    
    X_train_semi = []
    Y_train_semi = []
    
    for o in index:
        X_train_semi.append(X_train_semi_prime[o])
        Y_train_semi.append(Y_train_semi_prime[o])
    
    X_train_semi = np.asarray(X_train_semi)
    Y_train_semi = np.asarray(Y_train_semi)
    
    # print(X_train_semi,Y_train_semi)
    print("reliable unlabelled data:",Y_train_semi)
    X_train_semi_prime = np.delete(X_train_semi_prime, index, axis=0)
    Y_train_semi_prime = np.delete(Y_train_semi_prime, index, axis=0)
    print("data need to be predicted：", Y_train_semi_prime)
    X_train = np.concatenate((X_train, X_train_semi), axis=0) # 靠谱数据与train连接
    Y_train = np.concatenate((Y_train, Y_train_semi), axis=0)
    # print('Y_train_semi[10000]', Y_train_semi[10000])
    Y_train = ps.to_categorical(Y_train, nb_classes) # 按照最大标签概率确定类别
    # print('after ps Y_train_semi[10000]', Y_train_semi[10000])
    print('X_train_semi.shape', X_train.shape, 'Y_train_semi.shape', Y_train.shape)

    print(" === Training with unlabeled data...... === \n")
    # X_validation_semi, Y_validation_semi = ps.parseValidation(X_train, Y_train, nb_classes, len(X_train)*validPercent/100, _type='rgb')
#    sgd = SGD(lr=0.0001,decay=1e-6,momentum=0.9,nesterov=True)
#    model.compile(optimizer=sgd,loss="categorical_crossentropy",metrics=["accuracy"])
    history = model.fit(X_train, Y_train, batch_size=batch_size2, nb_epoch=nb_epoch2, validation_data=(X_test, Y_test), shuffle=True)
#    print(" === Saving model...... === \n")
#    tmp = MODEL_FILE + '_ver_' + str(i)
#    model.save(tmp)
#    model = load_model(tmp)
    
    
   # plt.plot(history.history['acc'])
#    plt.plot(history.history['val_acc'])
#    plt.ylim((0,1.05))
#    plt.title('model accuracy')
#    plt.ylabel('accuracy')
#    plt.xlabel('epoch')
#    plt.legend(['test'],loc='upper left')
#    my_y_ticks = np.arange(0,1,0.05)
#    plt.yticks(my_y_ticks)
#    plt.savefig("select_accuracy"+str(i)+".jpg")
#    plt.show()

    Y_pred_test_pre = model.predict(X_test)
    Y_pred_test = Y_pred_test_pre[:,1]
#    for i in range(66):
#        if Y_pred_test[i]>0.5:
#            Y_pred_test[i]=1
#        else:
#            Y_pred_test[i]=0
    Y_test_pos = Y_test[:,1]
    
#    tp = 0
#    fp = 0
#    fn = 0
#    tn = 0
#
#    for i in range(len(Y_test_pos)):
#        if Y_test_pos[i]==1 and Y_pred_test[i]==1:
#            tp=tp+1
#        if Y_test_pos[i]==1 and Y_pred_test[i]==0:
#            fn=fn+1
#        if Y_test_pos[i]==0 and Y_pred_test[i]==0:
#            tn=tn+1
#        if Y_test_pos[i]==0 and Y_pred_test[i]==1:
#            fp=fp+1
#    
#    
#    print("sensity:{},specificity:{}".format(tp/(tp+fn),tn/(fp+tn)))
    # print("!!!",Y_pred_test)
    fpr_model,tpr_model,thresholds_model = roc_curve(Y_test_pos,Y_pred_test)
    auc_model = auc(fpr_model,tpr_model) 
    plt.figure(1)
    plt.plot([0,1],[0,1],'k--')
    #plt.plot(fpr_model,tpr_model,label='semi-supervised(area={:.3f})'.format(auc_model))
    plt.plot(fpr_model,tpr_model,label='Semi-supervised')
    plt.xlabel('1-Specificity')
    plt.ylabel('Sensitivity')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.savefig("select_kf_2ROC"+str(i)+".eps")
    plt.show()
    

    if X_train_semi_prime.size==0:
        break
    else:
        print("continue:",i)
        
loss, accuracy = model.evaluate(X_test, Y_test)
print("the loss is:",loss,"the accuracy is:",accuracy)

# save model
print(" === Saving model...... === \n")
model.save(MODEL_FILE)


