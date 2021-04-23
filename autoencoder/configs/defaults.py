from yacs.config import CfgNode as CN

cfg = CN()

cfg.MODEL = CN()
cfg.IMAGE_SIZE = [300, 300]
cfg.ENCODING_SIZE = 32

cfg.ENCODER = CN()
cfg.ENCODER.FC_OUT_FEATURES = [128, 64]
cfg.ENCODER.CNV_OUT_CHANNELS = [32, 64, 128]

cfg.DECODER = CN()
cfg.DECODER.FC_OUT_FEATURES = [64, 128]
cfg.DECODER.CNV_OUT_CHANNELS = [128, 64, 32]

cfg.SOLVER = CN()
# train configs
cfg.SOLVER.MAX_ITER = 120000
cfg.SOLVER.N_SCHEDULE_STEPS = 240
cfg.SOLVER.GAMMA = 0.1
cfg.SOLVER.BATCH_SIZE = 32
cfg.SOLVER.LR = 1e-3
cfg.SOLVER.MOMENTUM = 0.9
cfg.SOLVER.WEIGHT_DECAY = 5e-4
cfg.SOLVER.DO_USE_LR_DECAY = True
cfg.SOLVER.SCHEDULER_STEPSIZE = 500
cfg.SOLVER.WARMUP_UNTIL = 500
cfg.SOLVER.WARMUP = False
cfg.SOLVER.WHICH_OPTIMIZER = "SGD" # Can be "Adam"

cfg.TEST.BATCH_SIZE = 10
cfg.EVAL_STEP = 2000 # Evaluate dataset every eval_step, disabled when eval_step < 0
cfg.MODEL_SAVE_STEP = 500 # Save checkpoint every save_step
cfg.LOG_STEP = 10 # Print logs every log_stepPrint logs every log_step
cfg.OUTPUT_DIR = "autoencoder_outputs"