from utils.torch_utils import get_device
from autoencoder.models.autoencoder import Autoencoder
from data_management.checkpoint import CheckPointer
import logging
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image

def do_train(
        cfg,
        model,
        train_loader,
        optimizer,
        checkpointer,
        arguments
    ):
    logger = logging.getLogger("SSD.trainer")
    logger.info("Start training ...")

def start_train(cfg):
    logger = logging.getLogger('SSD.trainer')

    model = Autoencoder(cfg=cfg)
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

    optimizer = optimizers[cfg.SOLVER.WHICH_OPTIMIZER]()
    dataset = tv_datasets[cfg.DATASET_NAME]

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    if cfg.DATASET_NAME == "coco":
        trainset = datasets.CocoDetection(
            root='./data',
            train=True,
            download=True,
            transform=transform
        )
        testset = datasets.CocoDetection(
            root='./data',
            train=False,
            download=True,
            transform=transform
        )
    trainloader = DataLoader(
        trainset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True
    )
    testloader = DataLoader(
        testset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True
    )

    arguments = {"iteration": 0}
    save_to_disk = True
    checkpointer = CheckPointer(
        model, optimizer, cfg.OUTPUT_DIR, save_to_disk, logger,
    )
    extra_checkpoint_data = checkpointer.load()
    arguments.update(extra_checkpoint_data)

    max_iter = cfg.SOLVER.MAX_ITER
    train_loader = make_data_loader(cfg, is_train=True, max_iter=max_iter, start_iter=arguments['iteration'])

    model = do_train(
        cfg, model, train_loader, optimizer,
        checkpointer, arguments)
    return model
    pass