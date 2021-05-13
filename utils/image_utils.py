from torchvision.utils import save_image
from .torch_utils import get_device
import numpy as np


MAX_FILTERS_TO_VISUALIZE = 10
def save_decod_img(img, epoch, cfg, w=None, h=None, range=None):
    if w is None or h is None:
        w = cfg.IMAGE_SIZE[0]
        h = cfg.IMAGE_SIZE[1]
    chan = img.size(1)
    if len(img.shape) > 2 and img.size(1) != cfg.IMAGE_CHANNELS:
        chan = [i for i in range(img.size(1))]
    # convolutional filter visualization
    if type(chan) == list:
        visualized = 0
        for r in np.rollaxis(img.detach().cpu().numpy(), 1):
            visualized += 1
            if visualized >= MAX_FILTERS_TO_VISUALIZE:
                break
            if get_device() == "cpu":
                pass
                #save_image(torch.Tensor(r), './autoenc_out/{}Autoencoder_image.png'.format(epoch))
    else:
        img = img.view(img.size(0), chan, w, h)
        save_image(img, './autoenc3_out/Autoencoder_image{}.png'.format(epoch), range=range)