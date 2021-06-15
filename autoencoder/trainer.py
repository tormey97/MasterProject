from utils.torch_utils import get_device
from autoencoder.models.autoencoder import Autoencoder
from autoencoder.models.gan import Network as GANEncoder
from autoencoder.models.test_enc import Autoencoder as at
from autoencoder.models.test_cnv import ConvAutoencoder
from attacker.eval import do_evaluate
from data_management.checkpoint import CheckPointer
from data_management.datasets.image_dataset import ImageDataset
from data_management.datasets.pascal_voc import VocDataset
from data_management.datasets.voc_detection import VOCDataset as VOCDetection
from vizer.draw import draw_boxes
from SSD.ssd.data.transforms import build_target_transform
import SSD.ssd.data.transforms as detection_transforms
import PIL as Image
import logging
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np
import cv2
from utils.entity_utils import create_target, create_bb_target, create_encoder
from utils.image_utils import save_decod_img
import os



def make_dir():
    image_dir = 'MNIST_Out_Images'
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

def visualize_layer_chain(recon_img, target_img, enc_outputs, dec_outputs, cfg):
    # TODO show activations
    pass

def do_train(
        cfg,
        model,
        data_loader,
        test_loader,
        optimizer,
        checkpointer,
        arguments,
        args,
        target_cfg=None,
        bb_target_cfg=None,
    ):

    if bb_target_cfg is not None:
        black_box_target = create_bb_target(bb_target_cfg)
    logger = logging.getLogger("SSD.trainer")
    logger.info("Start training ...")
    make_dir()
    model.train()
    criterion = torch.nn.MSELoss()
    if cfg.SOLVER.LOSS_FUNCTION == "BCE":
        criterion = torch.nn.BCELoss()

    if cfg.MODEL.MODEL_NAME == "gan" or cfg.MODEL.MODEL_NAME == "gan_object_detector":
        gen_optim = optimizer[0]
        disc_optim = optimizer[1]

    if cfg.MODEL.MODEL_NAME == "gan_object_detector":
        target = create_target(target_cfg)
        target.train()

    iteration = arguments["iteration"]
    for epoch in range(0, 10000):
        last_log = 0

        for i, (images, targets, _) in enumerate(data_loader):

            iteration += 1
            arguments["iteration"] = iteration

            # Convert to cuda
            images = images.to(get_device())
            targets["boxes"] = targets["boxes"].to(get_device())
            targets["labels"] = targets["labels"].to(get_device())

            # Run batch on model
            if cfg.MODEL.MODEL_NAME == "gan":
                reconstructed_images, encoding, quantized = model.encoder_generator(images)
                discriminator_rec = model.discriminator(reconstructed_images)
                discriminator_real = model.discriminator(images)


                disc_loss_real = torch.square(discriminator_real - 1.)
                disc_loss_rec = torch.square(discriminator_rec)

                disc_loss = torch.mean(disc_loss_real + disc_loss_rec)

                gen_loss = torch.mean(torch.square(discriminator_rec - 1.))
                distortion_penalty = 10 * torch.nn.MSELoss()(reconstructed_images, images)
                gen_loss += distortion_penalty

                gen_optim.zero_grad()
                disc_optim.zero_grad()

                gen_loss.backward(retain_graph=True)
                disc_loss.backward()

                gen_optim.step()
                disc_optim.step()
                print(gen_loss, disc_loss)
                if iteration % cfg.DRAW_STEP == 0:
                    save_decod_img(images.cpu().data, "TARGET" + str(iteration) + "gud", cfg)
                    save_decod_img(reconstructed_images.cpu().data, "RECONSTRUCTION" + str(epoch) + "_" + (str(iteration)),
                                   cfg)
                #generator loss:

            elif cfg.MODEL.MODEL_NAME == "gan_object_detector":
                images = torch.divide(images, 255)

                for j in range(cfg.SOLVER.ITERATIONS_PER_IMAGE):
                    gen_optim.zero_grad()
                    disc_optim.zero_grad()

                    perturbations, encoding, quantized = model.encoder_generator(images)
                    perturbations = torch.nn.functional.interpolate(perturbations, size=(300, 300), mode='bilinear')
                    perturbed_images = torch.add(perturbations, images)

                    loss_dict_original = target(images, targets=targets)
                    loss_dict_perturbed = target(perturbed_images, targets=targets)

                    discriminator_rec = torch.mean(model.discriminator(perturbed_images))

                    gen_loss = torch.multiply(torch.square(discriminator_rec - 1.), cfg.SOLVER.DISCRIMINATOR_IMPORTANCE)
                    true_labels = loss_dict_perturbed["labels"]
                    # want to give loss based on transforming labels != 0 to 10. so confidence in 10 where true label != 0
                    # should be higher.
                    labels_with_gt_nonzero = true_labels[true_labels > 0]
                    hard_neg_mask = loss_dict_original["mask"]
                    scores_original = torch.nn.functional.softmax(loss_dict_original["confidence"][0])
                    scores_perturbed = torch.nn.functional.softmax(loss_dict_perturbed["confidence"][0])
                    targeting = torch.zeros(scores_perturbed.shape)
                    if get_device() == "cuda":
                        targeting = targeting.cuda()
                    targeting[:, cfg.SOLVER.TARGET_CLASS] = 1
                    criterion = torch.nn.BCELoss()
                    object_hiding_loss_orig = criterion(scores_original, targeting)
                    object_hiding_loss_pert = criterion(scores_perturbed, targeting)
                    object_hiding_loss = torch.pow(cfg.SOLVER.TARGET_CHI, (-1 * cfg.SOLVER.TARGET_SENSITIVITY * (torch.subtract(object_hiding_loss_orig, object_hiding_loss_pert))))
                    cls_loss = cfg.SOLVER.CLS_LOSS_FACTOR * (loss_dict_perturbed["cls_loss"] - loss_dict_original["cls_loss"])
                    reg_loss = cfg.SOLVER.REG_LOSS_FACTOR * (loss_dict_perturbed["reg_loss"] - loss_dict_original["reg_loss"])
                    gen_loss += cfg.SOLVER.TARGET_LOSS_FACTOR * object_hiding_loss
                    performance_degradation_loss = cls_loss + reg_loss
                    gen_loss += cfg.SOLVER.PERFORMANCE_DEGRADATION_FACTOR * torch.pow(cfg.SOLVER.CHI, (-1 * performance_degradation_loss))

                    perturbation_mse = torch.nn.MSELoss()(torch.multiply(perturbed_images, 255), torch.multiply(images, 255))
                    hinge_loss = torch.nn.ReLU()(torch.subtract(perturbation_mse, cfg.SOLVER.HINGE_LOSS_THRESHOLD))

                    distortion_penalty = cfg.SOLVER.DISTORTION_PENALTY_FACTOR * perturbation_mse
                    gen_loss += distortion_penalty + cfg.SOLVER.HINGE_LOSS_FACTOR * hinge_loss


                    gen_loss.backward()
                    gen_optim.step()


                    discriminator_rec2 = model.discriminator(perturbed_images.detach())
                    discriminator_real = model.discriminator(images)

                    disc_loss_real = torch.square(discriminator_real - 1.)
                    disc_loss_rec = torch.multiply(torch.square(discriminator_rec2), cfg.SOLVER.DISC_REC_IMPORTANCE)

                    disc_loss = torch.add(torch.mean(disc_loss_real), torch.mean(disc_loss_rec))

                    disc_loss.backward()

                    disc_optim.step()

                    if iteration % cfg.LOG_STEP == 0:
                        logger.info("============================================================================")
                        logger.info("gen_loss: {gen_loss} \n disc_loss: {disc_loss} \n perf_degradation: {perf_deg}, obj_hiding: {object_hiding_loss} \n"
                                    " distortion_penalty: {distortion_penalty} \n, hinge_loss: {hinge_loss}, \n loss_dict_orig: {loss_dict_orig}"
                                    "\n loss dict perturbed: {loss_dict_perturbed} \n".format(gen_loss=gen_loss,
                                                                                              disc_loss=(disc_loss_rec, disc_loss_real),
                                                                                              perf_deg=performance_degradation_loss,
                                                                                              object_hiding_loss=object_hiding_loss,
                                                                                              distortion_penalty=distortion_penalty,
                                                                                              hinge_loss=hinge_loss,
                                                                                              loss_dict_orig="",
                                                                                              loss_dict_perturbed=""))


                    print(gen_loss, disc_loss)

                def draw_image2(image, name):
                    save_decod_img(image, str(iteration) + name, cfg=cfg, range=(0, 255))

                if iteration % cfg.DRAW_STEP == 0:
                    draw_image2(perturbed_images.cpu().data, "perturbed")
                    draw_image2(images.cpu().data, "original")
                    draw_image2(perturbations.cpu().data, "perturbations")

                def draw_image(image, name):
                    cv2image = image.detach().cpu().numpy()[0].transpose((1, 2, 0))
                    drawn_image = draw_boxes(cv2image, [], [], [],
                                             VOCDetection.class_names).astype(np.uint8)
                    Image.Image.fromarray(drawn_image).save(
                        os.path.join("autoenc2_out", str(iteration) + name + ".jpg"))



            elif cfg.MODEL.MODEL_NAME == "autoencoder":
                reconstructed_images, dec_outputs, enc_outputs = model(images)

                # -------- L1 Regularization -------- #
                added_loss = 0
                if cfg.SOLVER.L1_REGULARIZATION_FACTOR > 0:
                    l1_enc = sum([torch.norm(output) for output in enc_outputs])
                    #l1_dec = sum([torch.norm(output) for output in dec_outputs])
                    added_loss += cfg.SOLVER.L1_REGULARIZATION_FACTOR * (l1_enc)
                optimizer.zero_grad()

                # Calculate loss
                loss = criterion(reconstructed_images, images) + added_loss,
                loss[0].backward()

                optimizer.step()

                # Print
                if iteration == 0 or iteration - last_log >= cfg.LOG_STEP:
                    last_log = iteration
                    logger.info("ITERATION: ", iteration, "LOSS: ", loss, "ENC_LOSS: ", added_loss)
                    print("_ITERATION: ", iteration, "LOSS: ", loss, "ENC_LOSS: ", added_loss)

                    save_decod_img(images.cpu().data, "TARGET" + str(iteration) + "gud", cfg)
                    save_decod_img(reconstructed_images.cpu().data, "RECONSTRUCTION" + str(epoch)+"_"+(str(iteration)), cfg)
                    # Visualizing output features
                    for i in range(len(enc_outputs)):
                        pass
                        save_decod_img(enc_outputs[i],"ENCODING" + str(epoch) + "_" + str(iteration) + "_" + str(i) + "_" + "enc", cfg, w=enc_outputs[i].shape[2], h=enc_outputs[i].shape[3])
                        save_decod_img(dec_outputs[i],"DECODING" + str(epoch) + "_" +  str(iteration) + "_" + str(i) + "_" + "dec", cfg, w=dec_outputs[i].shape[2],
                                       h=dec_outputs[i].shape[3])

            if iteration % cfg.MODEL.SAVE_STEP == 0:
                print("SAVING MODEL AT ITERATION ", iteration)
                if cfg.MODEL.MODEL_NAME == "gan" or cfg.MODEL.MODEL_NAME == "gan_object_detector":
                    for i in checkpointer:
                        checkpointer[i].save("{}_{:06d}".format(i, iteration), **arguments)
                else:
                    checkpointer.save("model_{:06d}".format(iteration), **arguments)
            if iteration % cfg.MODEL.EVAL_STEP == 0:
                targets = dict(
                    white_box=target,
                    black_box=black_box_target,
                )
                result = do_evaluate(cfg, model, test_loader,
                            checkpointer, arguments, targets)


def start_train(cfg, target_cfg, bb_target_cfg):
    logger = logging.getLogger('SSD.trainer')
    models = {
        "autoencoder": Autoencoder,
        "gan": GANEncoder,
        "gan_object_detector": GANEncoder
    }
    model = models[cfg.MODEL.MODEL_NAME](cfg=cfg)
    model.train()
    model.to(get_device())


    optimizers = {
        "SGD": lambda: torch.optim.SGD(
            model.parameters(),
            lr=cfg.SOLVER.LR,
            momentum=cfg.SOLVER.MOMENTUM,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY
        ),
        "Adam": lambda: torch.optim.Adam(
            model.parameters(),
            lr=cfg.SOLVER.LR,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        )
    }

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(cfg.IMAGE_SIZE),
        transforms.RandomApply(
            [transforms.RandomCrop(cfg.IMAGE_SIZE[0] - 64)],
            p=0.4
        ),
        transforms.Resize(cfg.IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.3, 0.2, 0.6, 0.05),
        transforms.ToTensor(),
    ])
    trainset, testset = None, None

    if cfg.DATASET_NAME == "mnist_fashion":
        trainset = datasets.FashionMNIST(
            root='./datasetss',
            train=True,
            download=True,
            transform=transform
        )
        testset = datasets.FashionMNIST(
            root='./datasetss',
            train=False,
            download=True,
            transform=transform
        )
    elif cfg.DATASET_NAME == "test":
        trainset = ImageDataset(transform=transform)
        testset = ImageDataset(transform=transform)
    elif cfg.DATASET_NAME =="voc_detection":
        target_transform = build_target_transform(target_cfg)
        transform = detection_transforms.Compose([
            detection_transforms.Resize(target_cfg.INPUT.IMAGE_SIZE),
            detection_transforms.ConvertFromInts(),
            detection_transforms.ToTensor(),
        ])

        trainset = VOCDetection(
            data_dir='./datasets/Voc/VOCdevkit/VOC2012',
            transform=transform,
            split="train",
            target_transform=target_transform,
            keep_difficult=True
        )
        testset = VOCDetection(
            data_dir='./datasets/Voc/VOCdevkit/VOC2012',
            transform=transform,
            split="val",
            target_transform=target_transform,
            keep_difficult=True
        )
    elif cfg.DATASET_NAME == "voc":
        trainset = VocDataset(
            download=True,
            root='../datasets/Voc',
            transform=transform,
            image_set="train"
        )

        testset = VocDataset(
            download=True,
            root='../datasets/Voc',
            transform=transform,
            image_set="val"
        )
    elif cfg.DATASET_NAME == "coco":
        trainset = datasets.CocoDetection(
            root='../datasets/Coco',
            transform=transform,
            annFile='../datasets/Coco/annotations/instances_train2017.json'
        )
        testset = datasets.CocoDetection(
            root='../datasets/Coco',
            transform=transform,
            annFile='../datasets/Coco/annotations/instances_val2017.json'
        )
    train_loader = DataLoader(
        trainset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )
    testloader = DataLoader(
        testset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    arguments = {"iteration": 0}
    save_to_disk = True
    if cfg.MODEL.MODEL_NAME == "gan" or cfg.MODEL.MODEL_NAME == "gan_object_detector":
        disc_optim = torch.optim.Adam(params=model.discriminator.parameters(), lr=cfg.SOLVER.DISC_LR)
        gen_optim = torch.optim.Adam(params=model.encoder_generator.parameters(), lr=cfg.SOLVER.LR)
        checkpointer = {
            "discriminator": CheckPointer(
                model.discriminator, disc_optim, cfg.OUTPUT_DIR, save_to_disk, logger, last_checkpoint_name="disc_chkpt.txt"
            ),
            "generator": CheckPointer(
                model.encoder_generator, gen_optim, cfg.OUTPUT_DIR, save_to_disk, logger, last_checkpoint_name="gen_chkpt.txt"
            ),
            "encoder": CheckPointer (
                model.encoder_generator.encoder, save_dir=cfg.OUTPUT_DIR, save_to_disk=save_to_disk, logger=logger, last_checkpoint_name="encoder_chkpt.txt"
            )
        }
        optimizer = [disc_optim, gen_optim]
    else:
        optimizer = optimizers[cfg.SOLVER.WHICH_OPTIMIZER]()
        checkpointer = CheckPointer(
            model, optimizer, cfg.OUTPUT_DIR, save_to_disk, logger,
        )
    if cfg.MODEL.MODEL_NAME != "gan" and cfg.MODEL.MODEL_NAME != "gan_object_detector":
        extra_checkpoint_data = checkpointer.load()
    else:
        extra_checkpoint_data = checkpointer["discriminator"].load()
        print(extra_checkpoint_data)
        checkpointer["generator"].load()
    arguments.update(extra_checkpoint_data)

    max_iter = cfg.SOLVER.MAX_ITER

    do_train(
        cfg, model, train_loader, testloader, optimizer,
        checkpointer, arguments, None, target_cfg, bb_target_cfg)
