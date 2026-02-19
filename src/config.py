import os
import random
import tensorflow as tf
import numpy as np

SEED = 42


def set_seeds(seed=SEED):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


WINDOW = 7  # selected based on preliminary tuning
HORIZON = 1  # one-step-ahead forecast
USE_LOG1P = True

# rolling-origin config
INITIAL_TRAIN_SIZE = 500
TEST_BLOCK_SIZE = 50
VAL_SIZE = 100
PLOT_FOLD = 0
