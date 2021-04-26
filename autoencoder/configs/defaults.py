from yacs.config import CfgNode as CN

cfg = CN()
cfg.MODEL = CN()
cfg.IMAGE_SIZE = [256, 256]
cfg.IMAGE_CHANNELS = 3
cfg.ENCODING_SIZE = 8

cfg.ENCODER = CN()
cfg.ENCODER.FC_OUT_FEATURES = [512, 256, 128, 64, 32, 16]
cfg.ENCODER.CNV_OUT_CHANNELS = [256, 128, 64, 32, 16, 8]

cfg.DECODER = CN()
cfg.DECODER.FC_OUT_FEATURES = [16, 32, 64, 128, 256, 512]
cfg.DECODER.CNV_OUT_CHANNELS = [16, 32, 64, 128, 256, 256]

cfg.SOLVER = CN()
# train configs
cfg.SOLVER.MAX_ITER = 120000
cfg.SOLVER.N_SCHEDULE_STEPS = 240
cfg.SOLVER.GAMMA = 0.1
cfg.SOLVER.BATCH_SIZE = 32
cfg.SOLVER.LR = 0.1
cfg.SOLVER.MOMENTUM = 0.9
cfg.SOLVER.WEIGHT_DECAY = 5e-4
cfg.SOLVER.DO_USE_LR_DECAY = True
cfg.SOLVER.SCHEDULER_STEPSIZE = 500
cfg.SOLVER.WARMUP_UNTIL = 500
cfg.SOLVER.WARMUP = False
cfg.SOLVER.WHICH_OPTIMIZER = "SGD" # Can be "Adam"
cfg.SOLVER.L1_REGULARIZATION_FACTOR = 0.00008

cfg.VISUALIZATION = CN()

cfg.VISUALIZATION.VISUALIZE_ITER = 50 # visualize every N iterations

cfg.TEST = CN()
cfg.TEST.BATCH_SIZE = 10
cfg.EVAL_STEP = 2000 # Evaluate dataset every eval_step, disabled when eval_step < 0
cfg.MODEL_SAVE_STEP = 500 # Save checkpoint every save_step
cfg.LOG_STEP = 5 # Print logs every log_stepPrint logs every log_step
cfg.OUTPUT_DIR = "autoencoder_outputs"
cfg.BATCH_SIZE = 8
cfg.DATASET_NAME = "voc"
