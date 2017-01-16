import sys
import os

homedir = os.path.expanduser('~')
sys.path.append(homedir + "/Workspace/models/slim")


from matplotlib import pyplot as plt

import numpy as np
import tensorflow as tf
import pandas as pd

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

with tf.Graph().as_default():
    # Decode string into matrix with intensity values
    sign_names = pd.read_csv('../SampleData/traffic-sign/signnames.csv')
    nb_classes = 43
    
    image = tf.image.decode_jpeg(tf.read_file('../SampleData/traffic-sign/stop.jpg'), channels=3)

    
    # Resize the input image, preserving the aspect ratio
    # and make a central crop of the resulted image.
    # The crop will be of the size of the default image size of
    # the network.
    processed_image = vgg_preprocessing.preprocess_image(image,
                                                         image_size,
                                                         image_size,
                                                         is_training=False)
    
    # Networks accept images in batches.
    # The first dimension usually represents the batch size.
    # In our case the batch size is one.
    processed_images  = tf.expand_dims(processed_image, 0)
    
    # Create the model, use the default arg scope to configure
    # the batch norm parameters. arg_scope is a very conveniet
    # feature of slim library -- you can define default
    # parameters for layers -- like stride, padding etc.
    with slim.arg_scope(resnet.resnet_arg_scope()):
        fc7 = resnet.ResNet(processed_images,
                            is_training=False,
                            feature_extract=True)
    shape = (fc7.get_shape().as_list()[-1], nb_classes)
    fc8W = tf.Variable(tf.truncated_normal(shape, stddev=1e-2))
    fc8b = tf.Variable(tf.zeros(nb_classes))
    logits = tf.nn.xw_plus_b(fc7, fc8W, fc8b)    

    # In order to get probabilities we apply softmax on the output.
    probabilities = tf.nn.softmax(logits)
    
    # Create a function that reads the network weights
    # from the checkpoint file that you downloaded.
    # We will run it in session later.
    init = tf.global_variables_initializer()
    init_fn = slim.assign_from_checkpoint_fn(
        os.path.join(checkpoints_dir, 'resnet_v1_50.ckpt'),
        slim.get_model_variables('resnet_v1_50'))
    
    with tf.Session() as sess:
        sess.run(init)

        # Load weights
        init_fn(sess)
        
        # We want to get predictions, image as numpy matrix
        # and resized and cropped piece that is actually
        # being fed to the network.
        np_image, network_input, probabilities = sess.run([image,
                                                           processed_image,
                                                           probabilities])
        probabilities = probabilities[0, 0:]
        sorted_inds = [i[0] for i in sorted(enumerate(-probabilities),
                                            key=lambda x:x[1])]
    
    # Show the image
    plt.figure()
    plt.imshow(np_image.astype(np.uint8))
    plt.suptitle("Original image", fontsize=14, fontweight='bold')
    plt.axis('off')
    
    # Show the image that is actually being fed to the network
    # The image was resized while preserving aspect ratio and then
    # cropped. After that, the mean pixel value was subtracted from
    # each pixel of that crop. We normalize the image to be between [-1, 1]
    # to show the image.
    plt.figure()
    plt.imshow( network_input / (network_input.max() - network_input.min()) )
    plt.suptitle("Resized, Cropped and Mean-Centered input to network",
                 fontsize=14, fontweight='bold')
    plt.axis('off')
    
    for i in range(5):
        index = sorted_inds[i]
        # Now we print the top-5 predictions that the network gives us with
        # corresponding probabilities. Pay attention that the index with
        # class names is shifted by 1 -- this is because some networks
        # were trained on 1000 classes and others on 1001. VGG-16 was trained
        # on 1000 classes.
        print('Probability %0.2f => [%s]' % (probabilities[index], sign_names.ix[index][1]))
        
    res = slim.get_model_variables()

    plt.show()
