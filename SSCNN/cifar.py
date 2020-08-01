# -*- coding:utf-8 -*-
import pickle,pprint
from PIL import Image
import numpy as np
import os
import matplotlib.image as plimg
import random

class DictSave(object):

    def __init__(self,filenames):
        self.filenames = filenames
        self.arr = []
        self.all_arr = []
        self.file_label = []
        self.file_labels = []
        self.names = []
        self.name = []
    def image_input(self,filenames):
        for filename in  filenames:
            self.arr,self.file_label = self.read_file(filename)
            if self.all_arr==[]:
                self.all_arr = self.arr
            else:
                self.all_arr = np.vstack([self.all_arr,self.arr])
            if self.file_labels == []:
                self.file_labels = self.file_label
            else:
                self.file_labels = np.concatenate((self.file_labels, self.file_label))
            if self.names == []:
                self.names = [filename]
            else:
                self.names = np.concatenate((self.names, [filename]))
        # print(self.names)
        # self.all_arr = np.split(self.all_arr,10,axis=0)
        # self.file_labels = np.split(self.file_labels, 10, axis=0)
    def read_file(self,filename):
        im = Image.open(os.path.join("./unlabelled/"+filename))#打开一个图像
        im = im.resize((224, 224))
        # 将图像的RGB分离
        r, g, b = im.split()
        # 将PILLOW图像转成数组
        r_arr = plimg.pil_to_array(r)
        g_arr = plimg.pil_to_array(g)
        b_arr = plimg.pil_to_array(b)
        # 将32*32二位数组转成1024的一维数组
        r_arr1 = r_arr.reshape(50176)
        g_arr1 = g_arr.reshape(50176)
        b_arr1 = b_arr.reshape(50176)
        # 标签
        # file_labels = []
        file_label = [0]
        if os.path.exists(os.path.join("Glaucoma+/"+filename)):
            file_label = [1]
            # print("Normal:",filename)
        # file_labels.append(file_label)
        # 3个一维数组合并成一个一维数组,大小为244860+1
        arr = np.concatenate((r_arr1, g_arr1, b_arr1))
        return arr,file_label
    def pickle_save(self,arr,file_labels,names):
        print("正在存储")
        # 构造字典,所有的图像诗句都在arr数组里,我这里是个以为数组,目前并没有存label
        # 在字典里存标签
        contact = {'label': file_labels,'data': arr,'names':names}
        # print(contact)
        f = open('label+_data', 'wb')
        pickle.dump(contact, f) # 把字典存到文本中去
        f.close()
        print("存储完毕")

    def pickle2_save(self,arr,file_labels):
        print("正在存储")
        # 构造字典,所有的图像诗句都在arr数组里,我这里是个以为数组,目前并没有存label
        # 在字典里存标签
        contact2 = {'label': file_labels, 'data': arr}
        f = open('test_data', 'wb')
        pickle.dump(contact2, f) # 把字典存到文本中去
        f.close()
        print("存储完毕")

    def pickle3_save(self,arr,names):
        print("正在存储")
        # 构造字典,所有的图像诗句都在arr数组里,我这里是个以为数组,目前并没有存label
        # 在字典里存标签
        # for i in range(10):
        contact3 = {'data': arr,'names':names}
        f = open('unlabelled_new', 'wb')
        pickle.dump(contact3, f) # 把字典存到文本中去
        f.close()
        print("存储完毕")

if __name__ == "__main__":
    # filenames = [os.path.join("Normal/", "Im%03d.bmp" %i ) for i in range(1, 169)] #100个图像
    # filenames = []
    # for i in range(1,414):
    #     filepath = os.path.join("all_label/", "Im%03d" %i)
    #     if os.path.exists(filepath):
    #         filenames = filenames.append(filepath)
    path = "./unlabelled"
    # 将文件夹下所有文件加入列表
    filenames = os.listdir(path)
    # random_test_list = list(random.sample(filenames, 66)) # 66张图片
    # for x in random_test_list:
    #     filenames.remove(x)
    # print(filenames)
    # print(filenames)
    # ds1 = DictSave(filenames)
    # print("train_image processing...")
    # ds1.image_input(ds1.filenames)
    # print("successfully image processing!")
    # # print(ds.file_labels)
    # ds1.pickle_save(ds1.all_arr,ds1.file_labels)
    #
    # ds2 = DictSave(random_test_list)
    # print("test_image processing...")
    # ds2.image_input(ds2.filenames)
    # ds2.pickle2_save(ds2.all_arr, ds2.file_labels)
    # print("successfully image processing!")
    # print("最终数组的大小:"+str(ds.all_arr.shape))

    # ds1 = DictSave(filenames)
    # print("train_image processing...")
    # ds1.image_input(ds1.filenames)
    # ds1.pickle_save(ds1.all_arr,ds1.file_labels,ds1.names)
    # print("successfully image processing!")

    ds3 = DictSave(filenames)
    print("unlabel image processing...")
    ds3.image_input(ds3.filenames)
    ds3.pickle3_save(ds3.all_arr,ds3.names)
    print("success")

# 划分数据集，66张测试，261张训练

# np.split(data, 10, axis=0)
