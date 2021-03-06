import torch
import utils.image_utils as img
from autoencoder.configs.defaults import cfg
class Perturber():
    def __call__(self, model, images):
        raise NotImplementedError


class GANPerturber():
    def __init__(self, network, image_size):
        self.network = network
        self.image_size = image_size
        self.i = 0

    def __call__(self, images, targets):
        self.i +=1
        images_div = torch.divide(images, 255)
        perturbations = list(self.network.encgen_forward(images))
        #perturbations[0] = torch.nn.functional.interpolate(perturbations[0], size=(images.shape[2], images.shape[3]), mode='bilinear')

        perturbed_images = torch.clip(torch.multiply(torch.add(perturbations[0], images_div), 255), 0, 255)

        #img.save_decod_img(torch.divide(perturbed_images, 255), str(self.i), cfg)
        return perturbed_images

