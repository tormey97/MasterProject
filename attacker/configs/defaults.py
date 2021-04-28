from yacs.config import CfgNode as CN

cfg = CN()
cfg.OUTPUT_DIR = "atk_outputs/output1"

cfg.REWARD = CN()
cfg.REWARD.DELTA_FACTOR = 0.1
cfg.REWARD.PERFORMANCE_REDUCTION_FACTOR = 15

cfg.TRAIN = CN()
cfg.TRAIN.MAX_EPISODES=10000