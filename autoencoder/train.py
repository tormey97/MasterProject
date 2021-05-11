import torch
from torch import nn
import argparse
from autoencoder.configs.defaults import cfg
import pathlib
from data_management.logger import setup_logger
from autoencoder.inference import do_evaluation
from autoencoder.trainer import start_train
from torchvision import datasets
from SSD.ssd.config.defaults import _C as target_cfg

def get_parser():
    parser = argparse.ArgumentParser(description='Autoencoder')
    parser.add_argument(
        "config_file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "target_config",
        default="",
        metavar="FILE",
        help="path to target config file",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


def main():
    args = get_parser().parse_args()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    target_cfg.merge_from_file(args.target_config)
    target_cfg.freeze()
    output_dir = pathlib.Path(cfg.OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True, parents=True)

    logger = setup_logger("Autoencoder", output_dir)
    logger.info(args)

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    model = start_train(cfg, target_cfg)

    logger.info('Start evaluating...')
    torch.cuda.empty_cache()  # speed up evaluating after training finished
    do_evaluation(cfg, model)


if __name__ == '__main__':
    main()

