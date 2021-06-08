from utils.torch_utils import get_device
from autoencoder.models.autoencoder import Autoencoder
from autoencoder.models.gan import Network as GANEncoder
from data_management.checkpoint import CheckPointer
from data_management.datasets.voc_detection import VOCDataset as VOCDetection
from vizer.draw import draw_boxes
from SSD.ssd.data.transforms import build_target_transform
import SSD.ssd.data.transforms as detection_transforms
import PIL as Image
import logging
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from utils.entity_utils import create_target, create_bb_target
import torch
from autoencoder.configs.defaults import cfg
import pathlib
from data_management.logger import setup_logger
from autoencoder.inference import do_evaluation
from SSD.ssd.engine.inference import (evaluate, _accumulate_predictions_from_multiple_gpus)
from SSD.ssd.config.defaults import _C as target_cfg
from SSD.ssd.structures.container import Container

from detectron2.config.defaults import _C as bb_target_cfg
import argparse
import numpy as np
from attacker.perturber import GANPerturber

def calculate_norms(images, perturbed_images):
    norms = [1, 2, float('inf')]
    difference = torch.subtract(perturbed_images, images)
    output_dict = {}
    for norm in norms:
        norm_val = torch.norm(images - perturbed_images, p=norm)
        output_dict[str(norm)] = norm_val.detach().cpu().numpy()

    output_dict["mse"] = torch.nn.MSELoss()(perturbed_images, images).detach().cpu().numpy()
    return output_dict


def compute_on_dataset(model, black_box_target, perturber, data_loader, device):

    def convert_output_format(output):
        output = output[0]["instances"]._fields
        container = Container(
            boxes=output["pred_boxes"].tensor,
            labels=torch.add(output["pred_classes"], 1),
            scores=output["scores"],
        )
        container.img_width = 300
        container.img_height = 300
        return [container]
    results_dict = {}
    results_dict_p = {}
    results_dict_bb = {}
    results_dict_bb_p = {}

    norm_dict = {}
    i = 0
    for batch in data_loader:
        i += 1
        if i == 6:
            break
        images, targets, image_ids = batch
        cpu_device = torch.device("cpu")
        with torch.no_grad():
            images = images.to(device)
            outputs = model(images)
            outputs_bb = black_box_target([{"image": images[0], "height": 300, "width": 300}])
            outputs_bb = convert_output_format(outputs_bb)
            perturbed_images = perturber(images, model)
            outputs_p = model(perturbed_images)
            outputs_bb_p = black_box_target([{"image": perturbed_images[0], "height": 300, "width": 300}])
            outputs_bb_p = convert_output_format(outputs_bb_p)
            norm_outputs = [calculate_norms(images, perturbed_images)]
            outputs = [o.to(cpu_device) for o in outputs]
            outputs_p = [o.to(cpu_device) for o in outputs_p]
            outputs_bb = [o.to(cpu_device) for o in outputs_bb]
            outputs_bb_p = [o.to(cpu_device) for o in outputs_bb_p]

        results_dict.update(
            {int(img_id): result for img_id, result in zip(image_ids, outputs)}
        )
        results_dict_p.update(
            {int(img_id): result for img_id, result in zip(image_ids, outputs_p)}
        )
        results_dict_bb.update(
            {int(img_id): result for img_id, result in zip(image_ids, outputs_bb)}
        )
        results_dict_bb_p.update(
            {int(img_id): result for img_id, result in zip(image_ids, outputs_bb_p)}
        )
        norm_dict.update(
            {int(img_id): result for img_id, result in zip(image_ids, norm_outputs)}
        )


    return results_dict, results_dict_p, results_dict_bb, results_dict_bb_p, norm_dict

def do_evaluate(cfg, model, testloader,
        checkpointer, arguments, target_cfg, bb_target_cfg):

    black_box_target = create_bb_target(bb_target_cfg)
    target = create_target(target_cfg)
    target.eval()
    perturber = GANPerturber(model)
    results, results_p, results_bb, results_bb_p, norm_dict = compute_on_dataset(target, black_box_target, perturber, testloader, get_device())

    eval_result = _accumulate_predictions_from_multiple_gpus(results)
    eval_result_p = _accumulate_predictions_from_multiple_gpus(results_p)
    eval_result_bb = _accumulate_predictions_from_multiple_gpus(results_bb)
    eval_result_bb_p = _accumulate_predictions_from_multiple_gpus(results_bb_p)

    norm_list = [norm_dict[i] for i in norm_dict.keys()]
    eval_result = evaluate(testloader.dataset, eval_result, "original_evals", norm_list)
    eval_result_p = evaluate(testloader.dataset, eval_result_p, "perturbation_evals", norm_list)
    eval_result_bb = evaluate(testloader.dataset, eval_result_bb, "bb_original_evals", norm_list)
    eval_result_bb_p = evaluate(testloader.dataset, eval_result_bb_p, "bb_perturbation_evals", norm_list)

    print(eval_result, "\n", eval_result_p, "\n", eval_result_bb, "\n", eval_result_bb_p)

def start_evaluation(cfg, target_cfg, bb_target_cfg):
    logger = logging.getLogger('SSD.trainer')
    models = {
        "autoencoder": Autoencoder,
        "gan": GANEncoder,
        "gan_object_detector": GANEncoder
    }
    model = models[cfg.MODEL.MODEL_NAME](cfg=cfg)
    model.train()
    model.to(get_device())
    target_transform = build_target_transform(target_cfg)
    transform = detection_transforms.Compose([
        detection_transforms.Resize(target_cfg.INPUT.IMAGE_SIZE),
        detection_transforms.ConvertFromInts(),
        detection_transforms.ToTensor(),
    ])
    testset = VOCDetection(
        data_dir='./datasets/Voc/VOCdevkit/VOC2012',
        transform=transform,
        split="val",
        target_transform=target_transform,
        keep_difficult=True
    )

    testloader = DataLoader(
        testset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    arguments = {"iteration": 0}
    save_to_disk = True
    if cfg.MODEL.MODEL_NAME == "gan" or cfg.MODEL.MODEL_NAME == "gan_object_detector":
        checkpointer = {
            "discriminator": CheckPointer(
                model.discriminator, None, cfg.OUTPUT_DIR, save_to_disk, logger, last_checkpoint_name="disc_chkpt.txt"
            ),
            "generator": CheckPointer(
                model.encoder_generator, None, cfg.OUTPUT_DIR, save_to_disk, logger, last_checkpoint_name="gen_chkpt.txt"
            ),
            "encoder": CheckPointer (
                model.encoder_generator.encoder, save_dir=cfg.OUTPUT_DIR, save_to_disk=save_to_disk, logger=logger, last_checkpoint_name="encoder_chkpt.txt"
            )
        }
    else:
        checkpointer = CheckPointer(
            model, None, cfg.OUTPUT_DIR, save_to_disk, logger,
        )
    if cfg.MODEL.MODEL_NAME != "gan" and cfg.MODEL.MODEL_NAME != "gan_object_detector":
        extra_checkpoint_data = checkpointer.load()
    else:
        #extra_checkpoint_data = checkpointer["discriminator"].load()
        #print(extra_checkpoint_data)
        checkpointer["generator"].load()
    #arguments.update(extra_checkpoint_data)

    max_iter = cfg.SOLVER.MAX_ITER

    model = do_evaluate(
        cfg, model, testloader,
        checkpointer, arguments, target_cfg, bb_target_cfg)
    return model


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
        "bb_target_config",
        default="",
        metavar="FILE",
        help="path to target black box config file",
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

    bb_target_cfg.merge_from_file(args.bb_target_config)
    output_dir = pathlib.Path(cfg.OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True, parents=True)

    logger = setup_logger("SSD.trainer", output_dir)
    logger.info(args)

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    model = start_evaluation(cfg, target_cfg, bb_target_cfg)

    logger.info('Start evaluating...')
    torch.cuda.empty_cache()  # speed up evaluating after training finished
    do_evaluation(cfg, model)

if __name__=="__main__":
    main()