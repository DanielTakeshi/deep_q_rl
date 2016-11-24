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


class HumanQNetwork:
    """ A Human Q-Network. """    

    def __init__(self):
        pass
