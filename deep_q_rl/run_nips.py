#! /usr/bin/env python
"""
Execute a training run of deep-Q-Leaning with parameters that
are consistent with:

Playing Atari with Deep Reinforcement Learning
NIPS Deep Learning Workshop 2013
"""

import launcher
import sys

class Defaults:
    # ----------------------
    # Experiment Parameters
    # ----------------------
    # Daniel: I decreased this to 40000 for first run to get results quicker.
    # Otherwise, that was the only change with the settings (aside from the
    # extra parameters due to the human net).
    STEPS_PER_EPOCH = 40000 
    EPOCHS = 100
    STEPS_PER_TEST = 10000

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
    UPDATE_RULE = 'rmsprop'
    BATCH_ACCUMULATOR = 'mean'
    LEARNING_RATE = .0002
    DISCOUNT = .95
    RMS_DECAY = .99 # (Rho)
    RMS_EPSILON = 1e-6
    MOMENTUM = 0
    CLIP_DELTA = 0
    EPSILON_START = 1.0
    EPSILON_MIN = .1
    EPSILON_DECAY = 1000000
    PHI_LENGTH = 4
    UPDATE_FREQUENCY = 1
    REPLAY_MEMORY_SIZE = 1000000
    BATCH_SIZE = 32
    NETWORK_TYPE = "nips_dnn"
    FREEZE_INTERVAL = -1
    REPLAY_START_SIZE = 100
    RESIZE_METHOD = 'crop'
    RESIZED_WIDTH = 84
    RESIZED_HEIGHT = 84
    DEATH_ENDS_EPISODE = 'false'
    MAX_START_NULLOPS = 0
    DETERMINISTIC = True
    CUDNN_DETERMINISTIC = False

    # ----------------------
    # Daniel's extra parameters:
    # ----------------------
    USE_HUMAN_DATA = False
    HUMAN_NET_PATH = "../human_nets/model_l1_0.0005_epochs_30_bsize_32.npz"
    # For now, to keep things simple, the epislon_min parameter from above
    # determines how many random actions. So with 0.1, for instance, we will
    # start out by playing 90% of actions determined from the human net, and
    # that ratio gradually decreases to 0% (while 90% will come from the
    # Q-learner). In other words, always leave 10% of actions for random
    # choices, which helps when certain actions are only supposed to be executed
    # O(1) times, such as FIRE in Breakout.

if __name__ == "__main__":
    launcher.launch(sys.argv[1:], Defaults, __doc__)
