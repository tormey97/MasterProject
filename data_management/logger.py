import logging
import os
import sys

# Taken from SSD implementation

def setup_logger(name, save_dir=None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # don't log results for the non-master process
    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    if save_dir:
        fh = logging.FileHandler(os.path.join(save_dir, 'log.txt'))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger
