#! /usr/bin/env python
"""
Execute a training run of deep-Q-Leaning with parameters that
are consistent with:

Human-level control through deep reinforcement learning.
Nature, 518(7540):529-533, February 2015

"""

import launcher
import sys

class Defaults:
    # ----------------------
    # Experiment Parameters
    # ----------------------
    STEPS_PER_EPOCH = 250000
    EPOCHS = 80 # changed from 200
    STEPS_PER_TEST = 125000

    # ----------------------
    # ALE Parameters
    # ----------------------
    BASE_ROM_PATH = "../roms/"
    ROM = 'breakout.bin'
    FRAME_SKIP = 4
    REPEAT_ACTION_PROBABILITY = 0

    # ----------------------
    # Agent/Network parameters:
    # ----------------------
    UPDATE_RULE = 'deepmind_rmsprop'
    BATCH_ACCUMULATOR = 'sum'
    LEARNING_RATE = .00025
    DISCOUNT = .99
    RMS_DECAY = .95 # (Rho)
    RMS_EPSILON = .01
    MOMENTUM = 0 # Note that the "momentum" value mentioned in the Nature
                 # paper is not used in the same way as a traditional momentum
                 # term.  It is used to track gradient for the purpose of
                 # estimating the standard deviation. This package uses
                 # rho/RMS_DECAY to track both the history of the gradient
                 # and the squared gradient.
    CLIP_DELTA = 1.0
    EPSILON_START = 1.0
    EPSILON_MIN = .1
    EPSILON_DECAY = 1000000*10 # 10 million
    PHI_LENGTH = 4
    UPDATE_FREQUENCY = 4
    REPLAY_MEMORY_SIZE = 1000000
    BATCH_SIZE = 32
    NETWORK_TYPE = "nature_dnn"
    FREEZE_INTERVAL = 10000
    REPLAY_START_SIZE = 50000
    RESIZE_METHOD = 'scale'
    RESIZED_WIDTH = 84
    RESIZED_HEIGHT = 84
    DEATH_ENDS_EPISODE = 'true'
    MAX_START_NULLOPS = 30
    DETERMINISTIC = True
    CUDNN_DETERMINISTIC = False

    # ----------------------
    # Daniel's extra parameters:
    # ----------------------
    USE_HUMAN_NET = False
    HUMAN_NET_PATH = "../human_nets/spaceinv_nature_model_l1_0.0005_epochs_30_bsize_32.npz"
    USE_HUMAN_EXPERIENCE_REPLAY = False
    HUMAN_EXPERIENCE_REPLAY_PATH = "../human_xp/breakout-human_experience_replay.npz"
    # For now, to keep things simple, the epislon_min parameter from above
    # determines how many random actions. So with 0.1, for instance, we will
    # start out by playing 90% of actions determined from the human net, and
    # that ratio gradually decreases to 0% (while 90% will come from the
    # Q-learner). In other words, always leave 10% of actions for random
    # choices, which helps when certain actions are only supposed to be executed
    # O(1) times, such as FIRE in Breakout.
    #
    # Update: I'm adding human experience replay. This also needs to be
    # annealed according to some formula.

if __name__ == "__main__":
    launcher.launch(sys.argv[1:], Defaults, __doc__)
