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


"""
# Draft of what code would look like
def create_human_qnet(model_file):

    with np.load(model_file) as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(network, param_values)

    # Return the output layer only (lasagne can find the rest from it).
    return network 
"""

if __name__ == "__main__":
    print("Don't call this directory. The methods should be imported elsewhere.")
