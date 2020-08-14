"""
With this script you can finetune AlexNet as provided in the alexnet.py
class on any given dataset.
Specify the configuration settings at the beginning according to your
problem.
This script was written for TensorFlow 1.0 and come with a blog post
you can find here:

https://kratzert.github.io/2017/02/24/finetuning-alexnet-with-tensorflow.html

Author: Frederik Kratzert
contact: f.kratzert(at)gmail.com
"""
import os
import numpy as np
import tensorflow as tf
from datetime import datetime
from alexnet import AlexNet
from datagenerator import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

"""
Configuration settings
"""

# Path to the textfiles for the trainings and validation set
train_file = './train.txt'
val_file = './val.txt'

# Learning params
learning_rate = 0.01
num_epochs = 50
batch_size = 32

# Network params
dropout_rate = 0.5
num_classes = 2
train_layers = ['fc8', 'fc7']

# How often we want to write the tf.summary data to disk
display_step = 1

# Path for tf.summary.FileWriter and to store model checkpoints
filewriter_path = "./file"
checkpoint_path = "./finetune_alexnet/"

# Create parent path if it doesn't exist
if not os.path.isdir(checkpoint_path): os.mkdir(checkpoint_path)

# TF placeholder for graph input and output
x = tf.placeholder(tf.float32, [batch_size, 227, 227, 3])
y = tf.placeholder(tf.float32, [None, num_classes])
keep_prob = tf.placeholder(tf.float32)

# Initialize model
model = AlexNet(x, keep_prob, num_classes, train_layers)

# Link variable to model output
score = model.fc8

# List of trainable variables of the layers we want to train
var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers]

# Op for calculating the loss
with tf.name_scope("cross_ent"):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=score, labels=y))

# Train op
with tf.name_scope("train"):
    # Get gradients of all trainable variables
    gradients = tf.gradients(loss, var_list)
    gradients = list(zip(gradients, var_list))

    # Create optimizer and apply gradient descent to the trainable variables
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.apply_gradients(grads_and_vars=gradients)

# Add gradients to summary
for gradient, var in gradients:
    tf.summary.histogram(var.name + '/gradient', gradient)

# Add the variables we train to the summary
for var in var_list:
    tf.summary.histogram(var.name, var)

# Add the loss to summary
tf.summary.scalar('cross_entropy', loss)

# Evaluation op: Accuracy of the model
with tf.name_scope("accuracy"):
    correct_pred = tf.equal(tf.argmax(score, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
with tf.name_scope("pred"):
    pred = tf.argmax(score, 1)
# Add the accuracy to the summary
tf.summary.scalar('accuracy', accuracy)

# tf.summary.scalar('val_accuracy', accuracy)
# Merge all summaries together
merged_summary = tf.summary.merge_all()

# Initialize the FileWriter
writer = tf.summary.FileWriter(filewriter_path)

# Initialize an saver for store model checkpoints
saver = tf.train.Saver()

# Initalize the data generator seperately for the training and validation set
train_generator = ImageDataGenerator(train_file,
                                     horizontal_flip=True, shuffle=True)
val_generator = ImageDataGenerator(val_file, shuffle=False)

# Get the number of training/validation steps per epoch
train_batches_per_epoch = np.floor(train_generator.data_size / batch_size).astype(np.int16)
val_batches_per_epoch = np.floor(val_generator.data_size / batch_size).astype(np.int16)

# Start Tensorflow session
with tf.Session() as sess:
    # Initialize all variables
    sess.run(tf.global_variables_initializer())

    # Add the model graph to TensorBoard
    writer.add_graph(sess.graph)

    # Load the pretrained weights into the non-trainable layer
    model.load_initial_weights(sess)

    print("{} Start training...".format(datetime.now()))
    print("{} Open Tensorboard at --logdir {}".format(datetime.now(),
                                                      filewriter_path))
    acc_train = []
    acc_val = []
    epochs = []
    # Loop over number of epochs
    for epoch in range(50):

        print("{} Epoch number: {}".format(datetime.now(), epoch + 1))

        step = 1

        while step < train_batches_per_epoch:

            # Get a batch of images and labels
            batch_xs, batch_ys = train_generator.next_batch(batch_size)

            # And run the training op
            sess.run(train_op, feed_dict={x: batch_xs,
                                          y: batch_ys,
                                          keep_prob: dropout_rate})

            # Generate summary with the current batch of data and write to file
            if step % display_step == 0:
                s = sess.run(merged_summary, feed_dict={x: batch_xs,
                                                        y: batch_ys,
                                                        keep_prob: 1.})
                writer.add_summary(s, epoch * train_batches_per_epoch + step)

            step += 1

        # Validate the model on the entire validation set
        print("{} Start validation".format(datetime.now()))
        test_acc = 0.
        test_count = 0

        for _ in range(val_batches_per_epoch):
            batch_tx, batch_ty = val_generator.next_batch(batch_size)
            acc = sess.run(accuracy, feed_dict={x: batch_tx,
                                                y: batch_ty,
                                                keep_prob: 1.})
            test_acc += acc
            test_count += 1

        test_acc /= test_count
        # writer.add_summary(test_acc, epoch)
        acc_val.append(test_acc)
        epochs.append(epoch)

        print("{} Validation Accuracy = {:.4f}".format(datetime.now(), test_acc))

        # Reset the file pointer of the image data generator
        val_generator.reset_pointer()
        train_generator.reset_pointer()
        # val_generator.reset_pointer()
        # train_generator.reset_pointer()
        # print("{} Saving checkpoint of model...".format(datetime.now()))

        # save checkpoint of the model
        # checkpoint_name = os.path.join(checkpoint_path, 'model_epoch' + str(epoch + 1) + '.ckpt')
        # save_path = saver.save(sess, checkpoint_name)

        # print("{} Model checkpoint saved at {}".format(datetime.now(), checkpoint_name))
    result = []
    y_t = []
    score2 = []
    val_generator.reset_pointer()
    train_generator.reset_pointer()

    train_layers = ['conv3', 'conv4', 'conv5', 'pool5', 'fc6', 'fc7', 'fc8']
    for epoch in range(100):

        print("{} Epoch number: {}".format(datetime.now(), epoch + 1))

        step = 1

        while step < train_batches_per_epoch:

            # Get a batch of images and labels
            batch_xs, batch_ys = train_generator.next_batch(batch_size)

            # And run the training op
            sess.run(train_op, feed_dict={x: batch_xs,
                                          y: batch_ys,
                                          keep_prob: dropout_rate})

            # Generate summary with the current batch of data and write to file
            if step % display_step == 0:
                s = sess.run(merged_summary, feed_dict={x: batch_xs,
                                                        y: batch_ys,
                                                        keep_prob: 1.})
                writer.add_summary(s, epoch * train_batches_per_epoch + step)

            step += 1

        # Validate the model on the entire validation set
        print("{} Start validation".format(datetime.now()))
        test_acc = 0.
        test_count = 0

        for _ in range(val_batches_per_epoch):
            batch_tx, batch_ty = val_generator.next_batch(batch_size)
            acc = sess.run(accuracy, feed_dict={x: batch_tx,
                                                y: batch_ty,
                                                keep_prob: 1.})
            test_acc += acc
            test_count += 1

        test_acc /= test_count
        # writer.add_summary(test_acc, epoch)
        acc_val.append(test_acc)
        epochs.append(epoch)

        print("{} Validation Accuracy = {:.4f}".format(datetime.now(), test_acc))

        # Reset the file pointer of the image data generator
        val_generator.reset_pointer()
        train_generator.reset_pointer()
        # val_generator.reset_pointer()
        # train_generator.reset_pointer()
        # print("{} Saving checkpoint of model...".format(datetime.now()))

        # save checkpoint of the model
        # checkpoint_name = os.path.join(checkpoint_path, 'model_epoch' + str(epoch + 1) + '.ckpt')
        # save_path = saver.save(sess, checkpoint_name)

        # print("{} Model checkpoint saved at {}".format(datetime.now(), checkpoint_name))
    with open('alex_acc.txt', 'w') as f:
        for i in acc_val:
            f.write(str(i) + '\n')

    plt.plot(acc_val)
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.title('model accuracy')
    plt.legend(loc='upper left')
    plt.show()
    for i in range(val_batches_per_epoch):
        batch_tx, batch_ty = val_generator.next_batch(batch_size)
        output = sess.run(pred, feed_dict={x: batch_tx,
                                           keep_prob: 1.})
        for (z, g) in zip(output, batch_ty):
            score2.append(z)
            y_t.append(np.argmax(g))

    with open("alex_roc.txt", 'w') as F:
        for (z, g) in zip(y_t, score2):
            F.write(str(z) + ' ' + str(g) + '\n')
    tp = 0
    fn = 0
    fp = 0
    tn = 0
    for (x, y) in zip(y_t, score2):
        if x == 1 and y == 1:
            tp = tp + 1
        if x == 1 and y == 0:
            fn = fn + 1
        if x == 0 and y == 0:
            tn = tn + 1
        if x == 0 and y == 1:
            fp = fp + 1
    print("alexnet_sensity:{}, specificity:{}".format(tp / (tp + fn), tn / (fp + tn)))
    font1 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 13,
             }
    fpr, tpr, threshold = roc_curve(y_t, score2)  ###计算真正率和假正率
    roc_auc = auc(fpr, tpr)  ###计算auc的值
    plt.figure()
    lw = 2
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='CNN-V curve (area = %0.5f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Specificity', fontsize=18)
    plt.ylabel('Sensitivity ', fontsize=18)
    plt.legend(loc="lower right", prop=font1)
    plt.show()
