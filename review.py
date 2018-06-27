Pro Tip: Visualizing this VGG16 model using Tensorboard can be extremely useful. Below I provide you with a snippet to convert .pb file into TF summary. After converting it, you can run tensorboard --logdir=. in the same directory to start Tensorboard and visualize the graph in your browser.

import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.python.util import compat

with tf.Session() as sess:
    model_filename ='saved_model.pb'
    with gfile.FastGFile(model_filename, 'rb') as f:
        data = compat.as_bytes(f.read())
        sm = saved_model_pb2.SavedModel()
        sm.ParseFromString(data)
        g_in = tf.import_graph_def(sm.meta_graphs[0].graph_def)

LOGDIR='.'
train_writer = tf.summary.FileWriter(LOGDIR)
train_writer.add_graph(sess.graph)


In order to make the most of few training examples, we can augment them via a number of random transformations.
The model would never see twice the exact same picture and this helps prevent overfitting and helps the model generalize better.
For example, to flip, rotate and shift, use numpy/scipy operations.
def modify_picture(image, label):

    # flip
    if np.random.rand() > 0.5:
        image = np.fliplr(image)
        label = np.fliplr(label)

    # rotate    
    if np.random.rand() > 0.5:
        max_angle = 5
        image = scipy.ndimage.interpolation.rotate(image, random.uniform(-max_angle, max_angle))
        label = scipy.ndimage.interpolation.rotate(label, random.uniform(-max_angle, max_angle))

    # shift
    if np.random.rand() > 0.5:
        max_zoom = 1.3
        image = scipy.ndimage.interpolation.shift(image, random.uniform(-1, 1))    
        label = scipy.ndimage.interpolation.shift(label, random.uniform(-1, 1))

    return image, label
and call this in gen_batch_function() as image, gt_image = modify_picture(image, gt_image).