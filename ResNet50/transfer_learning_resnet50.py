import sys
import os

homedir = os.path.expanduser('~')
sys.path.append(homedir + "/Workspace/models/slim")

import pickle
import time
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from datasets import imagenet
from nets import vgg
from preprocessing import vgg_preprocessing

import resnet

checkpoints_dir = 'checkpoints'

slim = tf.contrib.slim

# We need default size of image for a particular network.
# The network was trained on images of that size -- so we
# resize input image later in the code.
image_size = resnet.resnet_v1.default_image_size

# Load the traffic sign data
nb_classes = 43
epochs = 50
batch_size = 128

with open('../SampleData/traffic-sign/train.p', 'rb') as f:
    data = pickle.load(f)

X_train, X_val, y_train, y_val = train_test_split(data['features'], data['labels'], test_size=0.1, random_state=0)

with open('../SampleData/traffic-sign/test.p', 'rb') as f:
    data = pickle.load(f)
X_test, y_test = data['features'], data['labels']

features = tf.placeholder(tf.float32, (None, 32, 32, 3))
labels = tf.placeholder(tf.int64, None)
processed_images = tf.image.resize_images(features, (image_size, image_size))

# Create the model, use the default arg scope to configure
# the batch norm parameters. arg_scope is a very conveniet
# feature of slim library -- you can define default
# parameters for layers -- like stride, padding etc.
with slim.arg_scope(resnet.resnet_arg_scope()):
    fc7 = resnet.ResNet(processed_images,
                        is_training=False,
                        feature_extract=True)
fc7 = tf.stop_gradient(fc7)
shape = (fc7.get_shape().as_list()[-1], nb_classes) 
fc8W = tf.Variable(tf.truncated_normal(shape, stddev=1e-2))
fc8b = tf.Variable(tf.zeros(nb_classes))
logits = tf.nn.xw_plus_b(fc7, fc8W, fc8b)    

cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels)
loss_op = tf.reduce_mean(cross_entropy)
opt = tf.train.AdamOptimizer()
train_op = opt.minimize(loss_op, var_list=[fc8W, fc8b])

preds = tf.arg_max(logits, 1)
accuracy_op = tf.reduce_mean(tf.cast(tf.equal(preds, labels), tf.float32))  

# Create a function that reads the network weights
# from the checkpoint file that you downloaded.
# We will run it in session later.
init_op = tf.global_variables_initializer()
init_fn = slim.assign_from_checkpoint_fn(os.path.join(checkpoints_dir, 'resnet_v1_50.ckpt'),
                                         slim.get_model_variables('resnet_v1_50'))

def eval_on_data(X, y, sess):
    total_acc = 0
    total_loss = 0
    for offset in range(0, X.shape[0], batch_size):
        end = offset + batch_size
        X_batch = X[offset:end]
        y_batch = y[offset:end]

        loss, acc = sess.run([loss_op, accuracy_op], feed_dict={features: X_batch, labels: y_batch})
        total_loss += (loss * X_batch.shape[0])
        total_acc += (acc * X_batch.shape[0])

    return total_loss/X.shape[0], total_acc/X.shape[0]

with tf.Session() as sess:
    sess.run(init_op)

    # Load weights
    init_fn(sess)

    for i in range(epochs):
        # training
        X_train, y_train = shuffle(X_train, y_train)
        t0 = time.time()
        for offset in range(0, X_train.shape[0], batch_size):
            end = offset + batch_size
            sess.run(train_op, feed_dict={features: X_train[offset:end], labels: y_train[offset:end]})

        val_loss, val_acc = eval_on_data(X_val, y_val, sess)
        test_loss, test_acc = eval_on_data(X_test, y_test, sess)
        print("Epoch", i)
        print("Time: %.3f seconds" % (time.time() - t0))
        print("Validation Loss =", val_loss)
        print("Validation Accuracy =", val_acc)
        print("Test Loss =", val_loss)
        print("Test Accuracy =", val_acc) # Best accuracy: 0.987
        print("")  
