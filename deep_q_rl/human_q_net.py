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

import make_net

class HumanQNetwork:
    """ A Human Q-Network. """    

    def __init__(self, input_width, input_height, num_actions, num_frames,
                       batch_size, network_type, human_net_path, map_action_index):
        self.input_width = input_width
        self.input_height = input_height
        self.num_actions = num_actions
        self.num_frames = num_frames
        self.batch_size = batch_size

        self.network_type = network_type
        self.human_net_path = human_net_path
        self.map_action_index = map_action_index

        # Use same method to make net as DQN, but also establish weights.  I
        # think this works, only thing different from my other code is that it
        # doesn't have an 'input_var' at the first layer. IDK if that matters.
        # Be careful with the self.num_actions variable!!
        #
        # Also, to keep code as general as possible, use T.tensor.4 but also
        # change 'phi' to be 4-dimensional, i.e. add an extra array by wrapping
        # this in a list and converting to a numpy array.
        self.input_var = T.tensor4('inputs') 
        self.net = make_net.build_network(self.network_type, self.input_width,
                                          self.input_height, self.num_actions,
                                          self.num_frames, self.batch_size,
                                          human_net=True, input_var=self.input_var)
        self._establish_weights()

        # Once we have weights, I think the following code will work. Note that
        # spragnur's nets do not use the softmax at the end.
        self.predictions = lasagne.layers.get_output(self.net, deterministic=True)
        self.predict_fxn = theano.function([self.input_var], self.predictions)


    def predict_action_from_state(self, phi):
        """ Obtain predicted action given state. 
        
        These are appropriately filtered through the action dictionary. However,
        we have to filter once, and THEN spragnur's code in ale_experiment.py
        will do ANOTHER filter, e.g. with Breakout spragnur's filter is arr=[0 1
        3 4], so if we return a 2 here, which in my data is RIGHT, I will return
        a 2, then spragnur's code maps to arr[2]=2 and the ALE officially 'sees'
        3 for its action. In **my** data, 0=NOOP, 1=LEFT, 2=RIGHT.

        Args:
            phi: A (phi_length, width, height)-dimensional numpy array
            representing the state.

        Returns:
            An integer representing the action to take.
        """
        a_probabilities = self.predict_fxn( np.array([phi]) ) # ADD DIMENSION!!
        assert len(a_probabilities[0]) == 3
        return self.map_action_index[ np.argmax(a_probabilities) ]


    def _establish_weights(self):
        """ From the Lasagne tutorial. The net must be built first! """
        with np.load(self.human_net_path) as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(self.net, param_values)
