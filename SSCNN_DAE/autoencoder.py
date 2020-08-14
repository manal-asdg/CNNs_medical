from skimage import io,transform
import glob
import os
import sys
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import math
from tensorflow.python.framework import graph_util


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE-DEVICES"]=""    # Change to '0' to use tf.device("/gpu:0")

path='D:/disease/unlabelled/U/'
model_path='D:/re/model.ckpt'
w=200
h=200
c=3
def read_img(path):
    imgs=[]
    for im in glob.glob(path+'/*.jpg'):
        img=io.imread(im)
        img=transform.resize(img,(w,h))
        imgs.append(img)
    return np.asarray(imgs,np.float32)
data=read_img(path)
print(data.shape)
##num_example=data.shape[0]
##ratio=0.8
##s=np.int(num_example*ratio)
##x_train=data[:s]
##x_val=data[s:]

x=tf.placeholder(tf.float32,shape=[None,w,h,c],name='x')
y_=tf.placeholder(tf.float32,shape=[None,w,h,c],name='y_')
    


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=1/math.sqrt(shape[2]))
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.zeros(shape)
  return tf.Variable(initial)

# Define CNN and max_pool operations
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
# In this CNN Based AutoEncoder, I will compress 784-dimension images to 128 (4 x 4 x 8) dimensions
# The latent code can be used for many purpose

def inference(input_tensor,batch_size, train, regularizer):

    with tf.variable_scope('l1-conv1'):
        conv1_weights = tf.get_variable("weight",[batch_size,3,3,32],initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable("bias", [32], initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.conv2d((input_tensor+0.4*tf.random_normal([batch_size,200,200,3])), conv1_weights, strides=[1, 2, 2, 1], padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))
    with tf.variable_scope('l2-conv2'):
        conv2_weights = tf.get_variable("weight",[3,3,32,16],initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable("bias", [16], initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d((relu1+0.3*tf.random_normal([batch_size,100,100,32])), conv2_weights, strides=[1,2,2, 1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))


    with tf.variable_scope('l3-conv3'):
        conv3_weights = tf.get_variable("weight",[3,3,16,8],initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv3_biases = tf.get_variable("bias", [8], initializer=tf.constant_initializer(0.0))
        conv3 = tf.nn.conv2d((relu2+0.2*tf.random_normal([batch_size,50,50,16])), conv3_weights, strides=[1, 2, 2, 1], padding='SAME')
        relu3 = tf.nn.relu(tf.nn.bias_add(conv3, conv3_biases))

    with tf.variable_scope('dl3-conv4'):
        dconv3_weights = tf.get_variable("weight",[3,3,16,8],initializer=tf.truncated_normal_initializer(stddev=0.1))
        dconv3_biases = tf.get_variable("bias", [16], initializer=tf.constant_initializer(0.0))
        dconv3 = tf.nn.conv2d_transpose((relu3+0.1*tf.random_normal([batch_size,25,25,8])), dconv3_weights,strides=[1, 2, 2, 1], output_shape=[batch_size,50,50,16],padding='SAME' )
        drelu3 = tf.nn.relu(tf.nn.bias_add(dconv3, dconv3_biases))

    
    with tf.variable_scope('dl2-conv5'):
        dconv2_weights = tf.get_variable("weight",[3,3,32,16],initializer=tf.truncated_normal_initializer(stddev=0.1))
        dconv2_biases = tf.get_variable("bias", [32], initializer=tf.constant_initializer(0.0))
        dconv2 = tf.nn.conv2d_transpose(drelu3, dconv2_weights,strides=[1, 2, 2, 1], output_shape=[batch_size,100,100,32],padding='SAME' )
        drelu2 = tf.nn.relu(tf.nn.bias_add(dconv2, dconv2_biases))

    with tf.variable_scope('dl1-conv6'):
        dconv1_weights = tf.get_variable("weight",[3,3,3,32],initializer=tf.truncated_normal_initializer(stddev=0.1))
        dconv1_biases = tf.get_variable("bias", [3], initializer=tf.constant_initializer(0.0))
        dconv1 = tf.nn.conv2d_transpose(drelu2, dconv1_weights,strides=[1, 2, 2, 1],output_shape=[batch_size,200,200,3], padding='SAME' )
        drelu1 = tf.nn.relu(tf.nn.bias_add(dconv1, dconv1_biases))
    return relu3,drelu1
#-------------------------网络结束--------------------------------
batch_size=40
regularizer = tf.contrib.layers.l2_regularizer(0.0001)
relu3_,drelu1_ = inference(x,batch_size,False,regularizer)

b = tf.constant(value=1,dtype=tf.float32)
relu3_eval = tf.multiply(relu3_,b,name='relu3_eval')
c = tf.constant(value=1,dtype=tf.float32)
drelu1_eval = tf.multiply(drelu1_,c,name='drelu1_eval')

loss=tf.reduce_sum(tf.square(drelu1_-y_))
train_op=tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)




#定义一个函数，按批次取数据
def minibatches(inputs=None, targets=None, batch_size=None, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt]

#训练和测试数据
n_epoch=1700                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
Loss_list=[]
saver=tf.train.Saver()
sess=tf.Session()  
sess.run(tf.global_variables_initializer())
##with tf.Session() as sess: 
##    sess.run(tf.global_variables_initializer())
##    graph_def=tf.get_default_graph().as_graph_def()
##    output_graph_def=graph_util.convert_variables_to_constants(sess,graph_def,['layer1-conv1/weight','layer1-conv1/bias','layer2-conv2/weight','layer2-conv2/bias','layer3-conv3/weight','layer3-conv3/bias'])
##    with tf.gfile.GFile('D:/re/model.pb','wb') as f:
##        f.write(output_graph_def.SerializeToString())
for epoch in range(n_epoch):
    start_time = time.time()

    #training
    train_loss, n_batch = 0, 0
    for x_train_a, y_train_a in minibatches(data, data, batch_size, shuffle=True):
        _,err=sess.run([train_op,loss], feed_dict={x: x_train_a, y_: y_train_a})
        train_loss += err; n_batch += 1
    Loss_list.append(train_loss)
    if epoch%10==0:
        print("   train loss: %f" %(np.sum(train_loss)/ n_batch))
        print(epoch)
    
xx=range(n_epoch)
yy=Loss_list
plt.plot(xx,yy)
plt.show()
saver.save(sess,model_path)
sess.close()      




