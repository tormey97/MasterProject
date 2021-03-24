import glob
import os
import sys
import numpy as np
import cv2
from collections import deque
from keras.applications.xception import Xception
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.models import Model
import tensorflow as tf
import keras.backend.tensorflow_backend as backend
from keras.callbacks import TensorBoard
from threading import Thread
from tqdm import tqdm
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import random
import time
import math
# Rule-based vehicles that take photos in many different conditions, and save them.


# We need



if __name__ == "main":
    # create a vehicle
    # create camera for vehicle
    #connect to
    pass