"""
Import this code so that we can make different networks. This lets us load
weights in.

(c) 2016 by Daniel Seita
"""

import sys
import numpy as np
import theano
import theano.tensor as T
import lasagne


def build_nature_network(self, input_width, input_height, output_dim,
                         num_frames, batch_size):
    """
    Build a large network consistent with the DeepMind Nature paper.
    """
    from lasagne.layers import cuda_convnet

    l_in = lasagne.layers.InputLayer(
        shape=(None, num_frames, input_width, input_height)
    )

    l_conv1 = cuda_convnet.Conv2DCCLayer(
        l_in,
        num_filters=32,
        filter_size=(8, 8),
        stride=(4, 4),
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.HeUniform(), # Defaults to Glorot
        b=lasagne.init.Constant(.1),
        dimshuffle=True
    )

    l_conv2 = cuda_convnet.Conv2DCCLayer(
        l_conv1,
        num_filters=64,
        filter_size=(4, 4),
        stride=(2, 2),
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.HeUniform(),
        b=lasagne.init.Constant(.1),
        dimshuffle=True
    )

    l_conv3 = cuda_convnet.Conv2DCCLayer(
        l_conv2,
        num_filters=64,
        filter_size=(3, 3),
        stride=(1, 1),
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.HeUniform(),
        b=lasagne.init.Constant(.1),
        dimshuffle=True
    )

    l_hidden1 = lasagne.layers.DenseLayer(
        l_conv3,
        num_units=512,
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.HeUniform(),
        b=lasagne.init.Constant(.1)
    )

    l_out = lasagne.layers.DenseLayer(
        l_hidden1,
        num_units=output_dim,
        nonlinearity=None,
        W=lasagne.init.HeUniform(),
        b=lasagne.init.Constant(.1)
    )

    return l_out


def build_nature_network_dnn(self, input_width, input_height, output_dim,
                             num_frames, batch_size):
    """
    Build a large network consistent with the DeepMind Nature paper.
    """
    from lasagne.layers import dnn

    l_in = lasagne.layers.InputLayer(
        shape=(None, num_frames, input_width, input_height)
    )

    l_conv1 = dnn.Conv2DDNNLayer(
        l_in,
        num_filters=32,
        filter_size=(8, 8),
        stride=(4, 4),
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.HeUniform(),
        b=lasagne.init.Constant(.1)
    )

    l_conv2 = dnn.Conv2DDNNLayer(
        l_conv1,
        num_filters=64,
        filter_size=(4, 4),
        stride=(2, 2),
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.HeUniform(),
        b=lasagne.init.Constant(.1)
    )

    l_conv3 = dnn.Conv2DDNNLayer(
        l_conv2,
        num_filters=64,
        filter_size=(3, 3),
        stride=(1, 1),
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.HeUniform(),
        b=lasagne.init.Constant(.1)
    )

    l_hidden1 = lasagne.layers.DenseLayer(
        l_conv3,
        num_units=512,
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.HeUniform(),
        b=lasagne.init.Constant(.1)
    )

    l_out = lasagne.layers.DenseLayer(
        l_hidden1,
        num_units=output_dim,
        nonlinearity=None,
        W=lasagne.init.HeUniform(),
        b=lasagne.init.Constant(.1)
    )

    return l_out


def build_nips_network(self, input_width, input_height, output_dim,
                       num_frames, batch_size):
    """
    Build a network consistent with the 2013 NIPS paper.
    """
    from lasagne.layers import cuda_convnet
    l_in = lasagne.layers.InputLayer(
        shape=(None, num_frames, input_width, input_height)
    )

    l_conv1 = cuda_convnet.Conv2DCCLayer(
        l_in,
        num_filters=16,
        filter_size=(8, 8),
        stride=(4, 4),
        nonlinearity=lasagne.nonlinearities.rectify,
        #W=lasagne.init.HeUniform(c01b=True),
        W=lasagne.init.Normal(.01),
        b=lasagne.init.Constant(.1),
        dimshuffle=True
    )

    l_conv2 = cuda_convnet.Conv2DCCLayer(
        l_conv1,
        num_filters=32,
        filter_size=(4, 4),
        stride=(2, 2),
        nonlinearity=lasagne.nonlinearities.rectify,
        #W=lasagne.init.HeUniform(c01b=True),
        W=lasagne.init.Normal(.01),
        b=lasagne.init.Constant(.1),
        dimshuffle=True
    )

    l_hidden1 = lasagne.layers.DenseLayer(
        l_conv2,
        num_units=256,
        nonlinearity=lasagne.nonlinearities.rectify,
        #W=lasagne.init.HeUniform(),
        W=lasagne.init.Normal(.01),
        b=lasagne.init.Constant(.1)
    )

    l_out = lasagne.layers.DenseLayer(
        l_hidden1,
        num_units=output_dim,
        nonlinearity=None,
        #W=lasagne.init.HeUniform(),
        W=lasagne.init.Normal(.01),
        b=lasagne.init.Constant(.1)
    )

    return l_out


def build_nips_network_dnn(self, input_width, input_height, output_dim,
                           num_frames, batch_size):
    """
    Build a network consistent with the 2013 NIPS paper.
    """
    # Import it here, in case it isn't installed.
    from lasagne.layers import dnn

    l_in = lasagne.layers.InputLayer(
        shape=(None, num_frames, input_width, input_height)
    )


    l_conv1 = dnn.Conv2DDNNLayer(
        l_in,
        num_filters=16,
        filter_size=(8, 8),
        stride=(4, 4),
        nonlinearity=lasagne.nonlinearities.rectify,
        #W=lasagne.init.HeUniform(),
        W=lasagne.init.Normal(.01),
        b=lasagne.init.Constant(.1)
    )

    l_conv2 = dnn.Conv2DDNNLayer(
        l_conv1,
        num_filters=32,
        filter_size=(4, 4),
        stride=(2, 2),
        nonlinearity=lasagne.nonlinearities.rectify,
        #W=lasagne.init.HeUniform(),
        W=lasagne.init.Normal(.01),
        b=lasagne.init.Constant(.1)
    )

    l_hidden1 = lasagne.layers.DenseLayer(
        l_conv2,
        num_units=256,
        nonlinearity=lasagne.nonlinearities.rectify,
        #W=lasagne.init.HeUniform(),
        W=lasagne.init.Normal(.01),
        b=lasagne.init.Constant(.1)
    )

    l_out = lasagne.layers.DenseLayer(
        l_hidden1,
        num_units=output_dim,
        nonlinearity=None,
        #W=lasagne.init.HeUniform(),
        W=lasagne.init.Normal(.01),
        b=lasagne.init.Constant(.1)
    )

    return l_out


def build_linear_network(self, input_width, input_height, output_dim,
                         num_frames, batch_size):
    """
    Build a simple linear learner.  Useful for creating
    tests that sanity-check the weight update code.
    """

    l_in = lasagne.layers.InputLayer(
        shape=(None, num_frames, input_width, input_height)
    )

    l_out = lasagne.layers.DenseLayer(
        l_in,
        num_units=output_dim,
        nonlinearity=None,
        W=lasagne.init.Constant(0.0),
        b=None
    )

    return l_out


if __name__ == "__main__":
    print("Don't call this directory. The methods should be imported elsewhere.")
