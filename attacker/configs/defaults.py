from yacs.config import CfgNode as CN

cfg = CN()
cfg.OUTPUT_DIR = "atk_outputs"
cfg.OUTPUT_FILE = "output1.zip"
cfg.REWARD = CN()
cfg.REWARD.DELTA_FACTOR = 0.00001
cfg.REWARD.PERFORMANCE_REDUCTION_FACTOR = 1500

cfg.TRAIN = CN()
cfg.TRAIN.MAX_EPISODES=10000
cfg.TRAIN.SAVE_STEP = 1000
cfg.TRAIN.SAVE_AMOUNT = 1000