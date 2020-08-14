
from skimage import io,transform
import glob
import os
import tensorflow as tf
import numpy as np
import time

w=200
h=200
c=3
tr=0
np.set_printoptions(threshold=np.inf)
for i in range(8):
    path = "D:/disease/relu/"+'{0}'.format(i)
    if i <=4:
        baocun='D:/disease/noise/Glaucoma/'
    else:
        baocun='D:/disease/noise/Normal/'

    def read_img(path):
        imgs=[]
        for im in glob.glob(path+'/*.jpg'):
            img=io.imread(im)
            img=transform.resize(img,(w,h))
            imgs.append(img)
        for im in glob.glob(path+'/*.bmp'):
            img=io.imread(im)
            img=transform.resize(img,(w,h))
            imgs.append(img)
        return np.asarray(imgs,np.float32)
    data=read_img(path)
    print(data.shape)
    
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('D:/disease/finalmodel(noise)/model.ckpt.meta')
        saver.restore(sess,tf.train.latest_checkpoint('D:/disease/finalmodel(noise)/'))
        graph = tf.get_default_graph()
        x = graph.get_tensor_by_name("x:0")
        feed_dict = {x:data}
        relu = graph.get_tensor_by_name("relu3_eval:0")
        relu3=sess.run(relu,feed_dict)
    for j in range(len(relu3)):
        np.save(baocun+'{0}.npy'.format(j+tr),relu3[j])
    tr=tr+40


    
    





    


  
