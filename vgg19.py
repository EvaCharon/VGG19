import tensorflow as tf
import numpy as np


class Vgg19:
    def __init__(self, x, keep_prob, num_classes, skip_layer,
                 weights_path='DEFAULT'):
        self.X = x
        self.NUM_CLASSES = num_classes
        self.KEEP_PROB = keep_prob
        self.SKIP_LAYER = skip_layer
        if weights_path == 'DEFAULT':
            self.WEIGHTS_PATH = 'vgg19.npy'
       

        self.create()

    def create(self):
        """
        load variable from npy to build the VGG
        """

        conv1_1 = convLayer(self.X, 3, 3, 1, 1, 64, "conv1_1" )
        conv1_2 = convLayer(conv1_1, 3, 3, 1, 1, 64, "conv1_2")
        pool1 = maxPoolLayer(conv1_2, 2, 2, 2, 2, "pool1")

        conv2_1 = convLayer(pool1, 3, 3, 1, 1, 128, "conv2_1")
        conv2_2 = convLayer(conv2_1, 3, 3, 1, 1, 128, "conv2_2")
        pool2 = maxPoolLayer(conv2_2, 2, 2, 2, 2, "pool2")

        conv3_1 = convLayer(pool2, 3, 3, 1, 1, 256, "conv3_1")
        conv3_2 = convLayer(conv3_1, 3, 3, 1, 1, 256, "conv3_2")
        conv3_3 = convLayer(conv3_2, 3, 3, 1, 1, 256, "conv3_3")
        conv3_4 = convLayer(conv3_3, 3, 3, 1, 1, 256, "conv3_4")
        pool3 = maxPoolLayer(conv3_4, 2, 2, 2, 2, "pool3")

        conv4_1 = convLayer(pool3, 3, 3, 1, 1, 512, "conv4_1")
        conv4_2 = convLayer(conv4_1, 3, 3, 1, 1, 512, "conv4_2")
        conv4_3 = convLayer(conv4_2, 3, 3, 1, 1, 512, "conv4_3")
        conv4_4 = convLayer(conv4_3, 3, 3, 1, 1, 512, "conv4_4")
        pool4 = maxPoolLayer(conv4_4, 2, 2, 2, 2, "pool4")

        conv5_1 = convLayer(pool4, 3, 3, 1, 1, 512, "conv5_1")
        conv5_2 = convLayer(conv5_1, 3, 3, 1, 1, 512, "conv5_2")
        conv5_3 = convLayer(conv5_2, 3, 3, 1, 1, 512, "conv5_3")
        conv5_4 = convLayer(conv5_3, 3, 3, 1, 1, 512, "conv5_4")
        pool5 = maxPoolLayer(conv5_4, 2, 2, 2, 2, "pool5")

        fcIn = tf.reshape(pool5, [-1, 7*7*512])
        fc6 = fcLayer(fcIn, 7*7*512, 4096, True, "fc6")
        dropout1 = dropout(fc6, self.KEEP_PROB)



        fc7 = fcLayer(dropout1, 4096, 4096, True, "fc7")
        dropout2 = dropout(fc7, self.KEEP_PROB)

        self.fc8 = fcLayer(dropout2, 4096, self.NUM_CLASSES, True, "fc8")
        
    def load_initial_weights(self, session):
        """Load weights from file into network.

        As the weights from http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/
        come as a dict of lists (e.g. weights['conv1'] is a list) and not as
        dict of dicts (e.g. weights['conv1'] is a dict with keys 'weights' &
        'biases') we need a special load function
        """
        # Load the weights into memory
        weights_dict = np.load(self.WEIGHTS_PATH, encoding='bytes').item()

        # Loop over all layer names stored in the weights dict
        for op_name in weights_dict:

            # Check if layer should be trained from scratch
            if op_name not in self.SKIP_LAYER:

                with tf.variable_scope(op_name, reuse=True):

                    # Assign weights/biases to their corresponding tf variable
                    for data in weights_dict[op_name]:

                        # Biases
                        if len(data.shape) == 1:
                            var = tf.get_variable('biases', trainable=False)
                            session.run(var.assign(data))

                        # Weights
                        else:
                            var = tf.get_variable('weights', trainable=False)
                            session.run(var.assign(data))

def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

def maxPoolLayer(x, kHeight, kWidth, strideX, strideY, name, padding = "SAME"):
    """max-pooling"""
    return tf.nn.max_pool(x, ksize = [1, kHeight, kWidth, 1],
                    strides = [1, strideX, strideY, 1], padding = padding, name = name)
def dropout(x, keepPro, name = None):
    """dropout"""
    return tf.nn.dropout(x, keepPro, name)    
    
    

def convLayer(x, kHeight, kWidth, strideX, strideY,
              featureNum, name, padding = "SAME"):
    """convlutional"""
    channel = int(x.get_shape()[-1])
    with tf.variable_scope(name) as scope:
        w = tf.get_variable("w", shape = [kHeight, kWidth, channel, featureNum])
        b = tf.get_variable("b", shape = [featureNum])
        featureMap = tf.nn.conv2d(x, w, strides = [1, strideY, strideX, 1], padding = padding)
        out = tf.nn.bias_add(featureMap, b)
        return tf.nn.relu(tf.reshape(out, featureMap.get_shape().as_list()), name = scope.name)
        
def fcLayer(x, inputD, outputD, reluFlag, name):
    """fully-connect"""
    with tf.variable_scope(name) as scope:
        w = tf.get_variable("w", shape = [inputD, outputD], dtype = "float")
        b = tf.get_variable("b", [outputD], dtype = "float")
        out = tf.nn.xw_plus_b(x, w, b, name = scope.name)
        if reluFlag:
            return tf.nn.relu(out)
        else:
            return out


