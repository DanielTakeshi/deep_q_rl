"""
Code to use a "Human Q-Network," which has the same functionality as the
Q-Network used in DQNs, except that we create this separately and load it in as
needed. Note that the code for loading isn't really part of the class (will
figure out a way to put it in another location).
"""

import sys

import lasagne
import numpy as np
import theano
import theano.tensor as T

"""
# Draft of what code would look like
def create_human_qnet(model_file):

    with np.load(model_file) as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(network, param_values)

    # Return the output layer only (lasagne can find the rest from it).
    return network 
"""


class HumanQNetwork:
    """ A Human Q-Network. """    

    def __init__(self):
        pass
