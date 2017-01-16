import sys
import os

homedir = os.path.expanduser('~')
sys.path.append(homedir + "/Workspace/models/slim")


from matplotlib import pyplot as plt
import scipy.misc
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle

from datasets import imagenet
from nets import resnet_v1

import py_selective_search as pyss

slim = tf.contrib.slim

checkpoints_dir = 'checkpoints'

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94

def mean_image_subtraction(image, means):
    image = image.astype(np.float32)
    for k, val in enumerate(means):
        image[:,:,k] -= val
    return image

def nms_detections(dets, overlap=0.3):
    """
    Non-maximum suppression: Greedily select high-scoring detections and
    skip detections that are significantly covered by a previously
    selected detection.

    This version is translated from Matlab code by Tomasz Malisiewicz,
    who sped up Pedro Felzenszwalb's code.

    Parameters
    ----------
    dets: ndarray
        each row is ['xmin', 'ymin', 'xmax', 'ymax', 'score']
    overlap: float
        minimum overlap ratio (0.3 default)

    Output
    ------
    dets: ndarray
        remaining after suppression.
    """
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    ind = np.argsort(dets[:, 4])

    w = x2 - x1
    h = y2 - y1
    area = (w * h).astype(float)

    pick = []
    while len(ind) > 0:
        i = ind[-1]
        pick.append(i)
        ind = ind[:-1]

        xx1 = np.maximum(x1[i], x1[ind])
        yy1 = np.maximum(y1[i], y1[ind])
        xx2 = np.minimum(x2[i], x2[ind])
        yy2 = np.minimum(y2[i], y2[ind])

        w = np.maximum(0., xx2 - xx1)
        h = np.maximum(0., yy2 - yy1)

        wh = w * h
        o = wh / (area[i] + area[ind] - wh)

        ind = ind[np.nonzero(o <= overlap)[0]]

    return dets[pick, :]

# We need default size of image for a particular network.
# The network was trained on images of that size -- so we
# resize input image later in the code.
image_size = resnet_v1.resnet_v1.default_image_size
names_list_filename = "names_list.p"
if os.path.exists(names_list_filename):
    names_list = pickle.load(open(names_list_filename,"rb"))
else:
    names = imagenet.create_readable_names_for_imagenet_labels()
    names_list = [names[k] for k in range(1,1001)]
    pickle.dump(names_list, open(names_list_filename,"wb"))

def main(image_filename):
    #image_filename = "../SampleData/poodle2.jpg"
    image = scipy.misc.imread(image_filename, mode='RGB')
    height, width, nchannels = image.shape
    boxes = pyss.get_windows(image_filename) # [xmin, ymin, xmax, ymax]
    boxes = np.array(boxes)
    n_boxes = len(boxes)

    # dilate (pad p pixels around the box)
    #p_dilate = 1
    #boxes[:,:2] = np.maximum(0, boxes[:,:2]-p_dilate)
    #boxes[:,2] = np.minimum(width, boxes[:,2]+p_dilate)
    #boxes[:,3] = np.minimum(height, boxes[:,3]+p_dilate)


    # divide into batches
    batch_size = 128
    batch_start = [x for x in range(0, n_boxes, batch_size)]
    batch_end = batch_start[1:] + [n_boxes]

    # crop and resize
    processed_images = np.zeros((n_boxes, image_size, image_size, nchannels))
    for k, box in enumerate(boxes):
        cropped_image = image[box[1]:box[3]+1, box[0]:box[2]+1, :]
        resized_image = scipy.misc.imresize(cropped_image, [image_size, image_size])
        processed_images[k] = mean_image_subtraction(resized_image, [_R_MEAN, _G_MEAN, _B_MEAN])

    with tf.Graph().as_default():
        images = tf.placeholder(tf.float32, (None, image_size, image_size, nchannels))
        
        # Create the model, use the default arg scope to configure
        # the batch norm parameters. arg_scope is a very conveniet
        # feature of slim library -- you can define default
        # parameters for layers -- like stride, padding etc.
        with slim.arg_scope(resnet_v1.resnet_arg_scope()):
            logits, _ = resnet_v1.resnet_v1_50(images,
                                               num_classes=1000,
                                               is_training=False)
        
        # In order to get probabilities we apply softmax on the output.
        probabilities = tf.nn.softmax(logits)
        
        # Create a function that reads the network weights
        # from the checkpoint file that you downloaded.
        # We will run it in session later.
        init_fn = slim.assign_from_checkpoint_fn(
            os.path.join(checkpoints_dir, 'resnet_v1_50.ckpt'),
            slim.get_model_variables('resnet_v1_50'))
        
        score = np.zeros((n_boxes, 1000))
        with tf.Session() as sess:
            
            # Load weights
            init_fn(sess)
            
            for start, end in zip(batch_start, batch_end):
                sys.stdout.write('\r{} / {}'.format(end, batch_end[-1]))
                sys.stdout.flush()
                score_per_run = sess.run(logits, feed_dict={images: processed_images[start:end]})
                score_per_run = sess.run(probabilities, feed_dict={images: processed_images[start:end]}) 
                score[start:end] = np.squeeze(score_per_run, (1,2))
            print()
        
        predictions_df = pd.DataFrame(score, columns=names_list)
        max_s = predictions_df.max(0)
        max_s.sort_values(ascending=False, inplace=True)
        print(max_s[:10])
        
        # Show the image
        plt.figure()
        plt.imshow(image)
        currentAxis = plt.gca()
        
        # Display the image with the boxes of top 5 score from selected class
        disp_idx = 2
        pred = predictions_df[max_s.index[disp_idx]].values
        dets = np.concatenate((boxes, pred[:,None]), axis=1)
        dets = nms_detections(dets, 10)
        for k in range(5):
            box = dets[k]
            print('box %d: %f' % (k, box[-1]))
            coords = (box[0], box[1]), box[2] - box[0], box[3] - box[1] 
            currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor='r', linewidth=1))
            currentAxis.text(box[0] + 3, box[1] + 3, str(k)+":"+ "%.2f" % box[-1], color='r')
        plt.title(max_s.index[disp_idx])
        
        plt.show()

if __name__ == "__main__":
    main(sys.argv[1])	
