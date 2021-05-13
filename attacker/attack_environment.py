import gym
import torch

import torch.nn.functional as F
import torch.utils.data
import os

from gym.spaces.box import Box


from SSD.ssd.data.datasets.evaluation.voc.eval_detection_voc import *

import SSD.ssd.data.transforms as detection_transforms
from SSD.ssd.data.transforms.transforms import Resize, ToCV2Image, ToTensor

from PIL import Image
from vizer.draw import draw_boxes
from SSD.ssd.data.datasets import COCODataset, VOCDataset

from utils.entity_utils import create_target, create_encoder
from utils.image_utils import save_decod_img


CONFIDENCE_THRESHOLD = 0.7 # TODO make configurable
IOU_THRESHOLD = 0.2


def calculate_iou(prediction_box, gt_box):
    """Calculate intersection over union of single predicted and ground truth box.

    Args:
        prediction_box (np.array of floats): location of predicted object as
            [xmin, ymin, xmax, ymax]
        gt_box (np.array of floats): location of ground truth object as
            [xmin, ymin, xmax, ymax]

        returns:
            float: value of the intersection of union for the two boxes.
    """

    xmin = 0
    ymin = 1
    xmax = 2
    ymax = 3

    intersection_height = min(gt_box[ymax], prediction_box[ymax]) - max(gt_box[ymin], prediction_box[ymin])
    intersection_width = min(gt_box[xmax], prediction_box[xmax]) - max(gt_box[xmin], prediction_box[xmin])

    intersection_width = max(0, intersection_width)
    intersection_height = max(0, intersection_height)
    intersection = intersection_width * intersection_height

    gt_box_area = (gt_box[xmax] - gt_box[xmin]) * (gt_box[ymax] - gt_box[ymin])
    prediction_box_area = (prediction_box[xmax] - prediction_box[xmin]) * (prediction_box[ymax] - prediction_box[ymin])
    union = gt_box_area + prediction_box_area - intersection


    # Compute union
    iou = intersection / union
    assert iou >= 0 and iou <= 1
    return iou

def get_all_box_matches(prediction_boxes: np.array, gt_boxes: np.ndarray, iou_threshold):
    """Finds all possible matches for the predicted boxes to the ground truth boxes.
        No bounding box can have more than one match.

        Remember: Matching of bounding boxes should be done with decreasing IoU order!

    Args:
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    Returns the matched boxes (in corresponding order):
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of box matches, 4].
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of box matches, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    """
    # Find all possible matches with a IoU >= iou threshold

    best_matches = np.zeros((gt_boxes.shape[0], 2))
    for (i, gt_box) in enumerate(gt_boxes):
        # find best predbox
        best = (-1, None)
        for (j, pred_box) in enumerate(prediction_boxes):
            iou_score = calculate_iou(pred_box, gt_box)
            if best[0] == -1 or best[1] < iou_score:
                best = (j, iou_score)
        best_matches[i][0] = best[0]
        best_matches[i][1] = best[1]
    all_matches = best_matches.copy()
    match_filter = best_matches[:, 1] >= iou_threshold
    best_matches = best_matches[best_matches[:, 1] >= iou_threshold]
    argsorted = best_matches[:, 1].argsort()
    matched_pred_boxes = prediction_boxes[best_matches[argsorted][:, 0].astype(int)]
    return matched_pred_boxes, gt_boxes[match_filter][argsorted], all_matches

MAX_FILTERS_TO_VISUALIZE = 10

class AttackEnvironment(gym.Env):
    def __init__(self, attacker_cfg, target_cfg, encoder_cfg, data_loader: torch.utils.data.DataLoader):
        super().__init__()
        self.attacker_cfg = attacker_cfg
        self.target_cfg = target_cfg
        self.encoder_cfg = encoder_cfg

        self.target = create_target(target_cfg)
        self.target.eval()
        self.encoder_decoder, self.optimizers = create_encoder(encoder_cfg)
        self.data_loader = data_loader

        self.dataset_iterable = enumerate(self.data_loader)

        # Current image and annotations/targets for that image
        self.image_data = None
        self.image = None
        self.annotations = None
        # Encoding of current image
        self.encoding = None
        self.encoding_pooling_output = None

        self.action_space = Box(-1, 1, [361])  # TODO configurable
        self.observation_space = Box(-1, 1, [361])

        self.step_ctr = 0

    def calculate_map(self, image, name):
        transform = detection_transforms.Compose([
            detection_transforms.ToCV2Image(),
            detection_transforms.ConvertFromInts(),
            ToTensor(),
        ])
        image_ = image[0]
        boxes = self.image_data[1]["boxes"][0]
        labels = self.image_data[1]["labels"][0]

        _, gt_boxes, gt_labels = (image_, self.annotations[1][0], self.annotations[1][1])

        image_, boxes, labels = transform(image_, boxes, labels)

        targets = {'boxes': boxes.detach().cpu().numpy(), 'labels': labels.detach().cpu().numpy()}
        preds = self.target(image_.unsqueeze(0), targets=targets)[0] #TODO device

        preds = preds.resize((self.img_info['width'], self.img_info['height']))

        pred_boxes = preds["boxes"].detach().cpu().numpy()
        pred_labels = preds["labels"].detach().cpu().numpy()
        pred_scores = preds["scores"].detach().cpu().numpy()
        indices = pred_scores > CONFIDENCE_THRESHOLD
        pred_boxes = pred_boxes[indices]
        pred_labels = pred_labels[indices]
        pred_scores = pred_scores[indices]

        # need to find out which boxes are the "same" and add the predictions to a dict

        # Quickly visualize image
        if self.step_ctr % 1 == 0:
            cv2image = image_.detach().cpu().numpy().transpose((1, 2, 0)).astype(np.uint8)
            drawn_image = draw_boxes(cv2image, pred_boxes, pred_labels, pred_scores, VOCDataset.class_names).astype(np.uint8)
            Image.fromarray(drawn_image).save(os.path.join("justtosee", str(self.step_ctr) + name + ".jpg"))

            def draw_image2(image, name2):
                save_decod_img(image.cpu().data, str(self.step_ctr) + name2, cfg=self.encoder_cfg, range=(0, 255))

            draw_image2(torch.divide(image, 255).cpu().data, name + "perturbedS")
        prec, rec = calc_detection_voc_prec_rec([pred_boxes],
                                                [pred_labels],
                                                [pred_scores],
                                                [gt_boxes],
                                                [gt_labels],
                                                None,
                                                iou_thresh=0.5)

        with torch.no_grad():
            ap = calc_detection_voc_ap(prec, rec, use_07_metric=False)


        # want to match each pred to a gt.
        _, _, matches = get_all_box_matches(pred_boxes, gt_boxes, iou_threshold=IOU_THRESHOLD)

        # want to reward based on how much the correct class was reduced in classification probability (pred_score)
        # so for each prediction box in the new, perturbed image: match each box to the ground truth.
        # then, get the score of the CORRECT label which might be some involved. E.g. if gt_label is Duck, get the output probability for duck
        # which might not be instant?

        # basic version:
        # get best match for each GT.
        good_matches = []
        for i, match in enumerate(matches):
            if match[1] > IOU_THRESHOLD:
                # if confidence is high:
                good_matches.append({"label": pred_labels[match[0].astype(np.uint8)], "score": match[1]})
            else:
                good_matches.append({"label": -1, "score": -1})  # match was not good enough to be considered by classifier

        return np.nan_to_num(ap).mean(), good_matches, pred_labels, pred_scores, gt_labels

    def calculate_class_reward(self, original_image, perturbed_image, perturbation):

        map_perturbed, perturbation_matches, perturbation_pred_labels, perturbation_pred_scores, _ = self.calculate_map(perturbed_image.detach(), "perturbed")
        map_orig, original_matches, original_pred_labels, original_pred_scores, gt_labels = self.calculate_map(original_image.detach(), "original")
        rewards = []
        for i in range(len(perturbation_matches)):
            perturbation_gt_match = perturbation_matches[i]
            original_gt_match = original_matches[i]
            original_match_label = original_gt_match["label"]
            perturbation_match_label = perturbation_gt_match["label"]
            gt_label = gt_labels[i]
            reward = 0
            if original_match_label != -1:
                if perturbation_match_label == -1:
                    # object hidden. give reward equal to original detection confidence score
                    reward = original_gt_match["score"]
                # original image found a good match with GT
                elif original_match_label == gt_label:
                    # original image had the right label. so the probability should go down
                    if perturbation_match_label != gt_label:
                        reward = perturbation_gt_match["score"]
                    else:
                        #perturbation classified same label, but maybe with less confidence
                        reward = original_gt_match["score"] - perturbation_gt_match["score"]

                else:
                    # original image had wrong label classified. so the probability should go up if they are the same
                    if perturbation_match_label == gt_label:
                        reward = 0  # perturbation caused a correct detection. low reward

                    else:
                        reward = perturbation_gt_match["score"]  # perturbation caused an incorrect classification

            else:
                if perturbation_match_label != -1:
                    # perturbation found something that was not in the orig img. so check if it is the right label
                    if perturbation_match_label == gt_label:
                        reward = 0
                        pass

                    else:
                        reward = perturbation_gt_match["score"]
                        pass
            rewards.append(reward)

        class_reward = sum(rewards)
        performance_reduction_factor = self.attacker_cfg.REWARD.PERFORMANCE_REDUCTION_FACTOR
        delta_factor = self.attacker_cfg.REWARD.DELTA_FACTOR
        diff = np.linalg.norm(perturbation)

        addon = 0
        if map_orig > 0 and map_orig - map_perturbed == 0:
            addon = -5
        reward = addon + performance_reduction_factor * (map_orig - map_perturbed)
        if self.step_ctr % 5 == 0:
            print("DIFF: ", diff)
            print("REWARD: ", reward)
            print("CLS_REWARD: ", class_reward)
            print("MAP_ORIG: ", map_orig)
            print("MAP_PERTURBED: ", map_perturbed)
        return class_reward + reward

    def apply_transformation(self, delta):

        delta = torch.nn.functional.interpolate(delta, size=(300, 300), mode='bilinear')
        perturbed_image = self.image + delta * 255
        return perturbed_image

    # override
    def step(self, action):
        # get perturbed encoding by applying action
        perturbed_encoding = action.reshape(1, 1, 19, 19)
        #perturbed_encoding = torch.nn.functional.interpolate(torch.Tensor(perturbed_encoding).reshape(1, 3, 10, 10), (19, 19))
        # decode the perturbed encoding to generate a transformation
        reconstruction, _ = self.encoder_decoder.decode(torch.Tensor([self.encoding]))
        perturbation_transformation, _ = self.encoder_decoder.decode(torch.Tensor(perturbed_encoding))

        perturbation_transformation = perturbation_transformation - reconstruction
        # perturb the current image
        perturbed_image = self.apply_transformation(perturbation_transformation)
        # calculate reward based on perturbed image
        class_reward = self.calculate_class_reward(self.image, perturbed_image, perturbation_transformation.detach().cpu().numpy())

        #perturbation_delta_loss = torch.nn.MSELoss()(self.image, perturbed_image).detach().numpy()
        #class_loss = torch.exp(torch.Tensor(-1 * class_reward))

        done = True  # Done is always true, we consider one episode as one image
        self.step_ctr += 1
        return perturbed_encoding.flatten(), class_reward, done, {}

    #override
    def reset(self):
        """
        :return: the initial state of the problem, which is an encoding of the image
        """
        i = 0
        try:
            i, values = next(self.dataset_iterable)

        except StopIteration:
            self.dataset_iterable = enumerate(self.data_loader)
            i, values = next(self.dataset_iterable)
        self.annotations = self.data_loader.dataset.get_annotation(i)
        self.img_info = self.data_loader.dataset.get_img_info(i)

        self.image_data = values
        self.image = values[0]
        self.encoding = self.encoder_decoder.encode(self.image)[0].detach().cpu().numpy()
        return self.encoding.flatten()


    #override
    def render(self, mode='human'):
        pass

    def close(self):
        pass

def create_env(attacker_config, encoder_config, target_config, data_loader):
    return AttackEnvironment(target_cfg=target_config, data_loader=data_loader, attacker_cfg=attacker_config, encoder_cfg=encoder_config)


