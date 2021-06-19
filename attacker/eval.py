from utils.torch_utils import get_device
from autoencoder.models.autoencoder import Autoencoder
from autoencoder.models.gan import Network as GANEncoder
from data_management.checkpoint import CheckPointer
from data_management.datasets.voc_detection import VOCDataset as VOCDetection
from data_management.datasets.coco_detection import COCODataset as COCODetection

from detectron2.model_zoo.model_zoo import get
from vizer.draw import draw_boxes
from SSD.ssd.data.transforms import build_target_transform
import SSD.ssd.data.transforms as detection_transforms
from PIL import Image
from attacker.defender import defender
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
from SSD.ssd.modeling.detector.ssd_detector import SSDDetector
from SSD.ssd.data.datasets import COCODataset, VOCDataset
import os
from detectron2.config.defaults import _C as bb_target_cfg
import argparse
import numpy as np
from attacker.perturber import GANPerturber

CONFIDENCE_THRESHOLD = 0.5

def calculate_norms(images, perturbed_images):
    norms = [1, 2, float('inf')]
    difference = torch.subtract(perturbed_images, images)
    output_dict = {}
    for norm in norms:
        norm_val = torch.norm(images - perturbed_images, p=norm)
        output_dict[str(norm)] = norm_val.detach().cpu().numpy()

    output_dict["mse"] = torch.nn.MSELoss()(perturbed_images, images).detach().cpu().numpy()
    return output_dict

def draw_detection_output(image, boxes, labels, scores, class_names, filename, folder_name):
    indices = scores > CONFIDENCE_THRESHOLD
    boxes = boxes[indices].cpu()
    labels = labels[indices].cpu()
    scores = scores[indices].cpu()
    cv2image = image.detach().clip(0, 255).cpu().numpy().transpose((1, 2, 0)).astype(np.uint8)
    drawn_image = draw_boxes(cv2image, boxes, labels, scores,
                             class_names).astype(
        np.uint8)
    Image.fromarray(drawn_image).save(os.path.join(folder_name, filename + ".jpg"))
def compute_on_dataset(target_models, perturber, data_loader, device, folder_name):

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

    defense_levels = [0, 5, 10, 25, 50, 100]

    def get_index(_t, _l):
        return _t + "_" + str(_l)
    result_dicts_original = {get_index(t, l): {} for t in target_models for l in defense_levels}
    result_dicts_perturbed = {get_index(t, l): {} for t in target_models for l in defense_levels}



    norm_dict = {}
    i = 0
    for batch in data_loader:
        i += 1
        images, targets, image_ids = batch
        cpu_device = torch.device("cpu")
        with torch.no_grad():
            images = images.to(device)
            perturbed_images = perturber(images, None)
            defended_images = defender(images, defense_levels)
            defended_perturbed_images = defender(perturbed_images, defense_levels)

            for t in target_models:
                for l in range(len(defense_levels)):
                    image = defended_images[l]
                    perturbed_image = defended_perturbed_images[l]
                    if defense_levels[l] == 0:
                        image = images
                        perturbed_image = perturbed_images
                    target_model = target_models[t]
                    model_input = [{"image": image[0], "height": 300, "width": 300}]
                    is_ssd = isinstance(target_model, SSDDetector)
                    if is_ssd:
                        model_input = image

                    output = target_model(model_input)

                    model_input = [{"image": perturbed_image[0], "height": 300, "width": 300}]
                    if is_ssd:
                        model_input = perturbed_image
                    output_perturbed = target_model(model_input)
                    if not is_ssd:
                        output = convert_output_format(output)
                        output_perturbed = convert_output_format(output_perturbed)

                    if i % 50 == 0:
                        draw_detection_output(
                            image=perturbed_image[0],
                            boxes=output_perturbed[0]["boxes"],
                            labels=output_perturbed[0]["labels"],
                            scores=output_perturbed[0]["scores"],
                            class_names=data_loader.dataset.class_names,
                            filename=str(i) + "_" + t + "_" + str(defense_levels[l]) + "_perturbed",
                            folder_name=folder_name
                        )

                        draw_detection_output(
                            image=image[0],
                            boxes=output[0]["boxes"],
                            labels=output[0]["labels"],
                            scores=output[0]["scores"],
                            class_names=data_loader.dataset.class_names,
                            filename=str(i) + "_" + t + "_" + str(defense_levels[l]) + "_original",
                            folder_name=folder_name
                        )

                    outputs_original = [o.to(cpu_device) for o in output]
                    outputs_perturbed = [o.to(cpu_device) for o in output_perturbed]
                    result_dicts_original[get_index(t, defense_levels[l])].update(
                        {int(img_id): result for img_id, result in zip(image_ids, outputs_original)}
                    )
                    result_dicts_perturbed[get_index(t, defense_levels[l])].update(
                        {int(img_id): result for img_id, result in zip(image_ids, outputs_perturbed)}
                    )

            norm_outputs = [calculate_norms(images, perturbed_images)]
            norm_dict.update(
                {int(img_id): result for img_id, result in zip(image_ids, norm_outputs)}
            )


    return result_dicts_original, result_dicts_perturbed, norm_dict

def do_evaluate(cfg, model, testloader,
        checkpointer, arguments, targets):

    for i in targets:
        targets[i].eval()

    perturber = GANPerturber(model)
    results, results_p, norm_dict = compute_on_dataset(targets, perturber, testloader, get_device(), cfg.DRAW_TO_DIR)
    norm_list = [norm_dict[i] for i in norm_dict.keys()]

    for i in results:
        result_original = results[i]
        result_perturbed = results_p[i]

        eval_result = _accumulate_predictions_from_multiple_gpus(result_original)
        eval_result_p = _accumulate_predictions_from_multiple_gpus(result_perturbed)

        eval_result = evaluate(testloader.dataset, eval_result, "original_evals", norm_list, i + "_" + cfg.DRAW_TO_DIR)
        eval_result_p = evaluate(testloader.dataset, eval_result_p, "perturbation_evals", norm_list, i + "_" + cfg.DRAW_TO_DIR)

        print(eval_result, "\n", eval_result_p, "\n")

    for i in targets:
        targets[i].train()
    model.train()

def eval_voc():
    pass
def eval_coco():
    pass
def start_evaluation(cfg, target_cfg, bb_target_cfg, dataset="voc"):
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

    if dataset == "voc":
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
    else:
        testset = COCODetection(
            data_dir='./datasets/Coco/val2017',
            ann_file='./datasets/Coco/annotations_trainval2017/annotations/instances_val2017.json',
            transform=transform,
            target_transform=None,
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

    if dataset == "coco":
        ssd_detector_configs = dict(

        )

        detectron_detector_configs = dict(
            X101_FPN='COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml',
            R101_FPN="COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml",
            R101_DC5="COCO-Detection/faster_rcnn_R_101_DC5_3x.yaml",
            R101_C4="COCO-Detection/faster_rcnn_R_101_C4_3x.yaml",
            R50_C4="COCO-Detection/faster_rcnn_R_50_C4_3x.yaml",
            RN_R50="COCO-Detection/retinanet_R_50_FPN_3x.yaml",
            RN_R101="COCO-Detection/retinanet_R_101_FPN_3x.yaml"
        )


        targets = {}
        for i in detectron_detector_configs:
            config = bb_target_cfg.clone()
            #config.merge_from_file(detectron_detector_configs[i])
            #targets[i] =  create_bb_target(config)
            targets[i] = get(detectron_detector_configs[i], trained=True)
        for i in ssd_detector_configs:
            config = target_cfg.clone()
            config.merge_from_file(ssd_detector_configs[i])
            targets[i] = create_target(config)
    else:
        config = target_cfg.clone()
        config.merge_from_file("./SSD/configs/efficient_net_b3_ssd300_voc0712_local.yaml")
        # targets[i] =  create_bb_target(config)
        targets = dict(
            white_box=create_target(target_cfg),
           # black_box=create_bb_target(bb_target_cfg),
            grey_box=create_target(config)
        )

    do_evaluate(
        cfg, model, testloader,
        checkpointer, arguments, targets)

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
        "dataset",
        default="voc",
        type=str
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

    model = start_evaluation(cfg, target_cfg, bb_target_cfg, args.dataset)

    logger.info('Start evaluating...')
    torch.cuda.empty_cache()  # speed up evaluating after training finished
    do_evaluation(cfg, model)

if __name__=="__main__":
    main()