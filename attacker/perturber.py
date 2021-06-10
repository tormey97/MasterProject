import torch
import utils.image_utils as img
from autoencoder.configs.defaults import cfg
class Perturber():
    def __call__(self, model, images):
        raise NotImplementedError


class GANPerturber():
    def __init__(self, network):
        self.network = network
        self.i = 0

    def __call__(self, images, targets):
        self.i +=1
        images_div = torch.divide(images, 255)
        perturbations = self.network(images)

        perturbed_images = torch.clip(torch.multiply(torch.add(perturbations[0], images_div), 255), 0, 10000)

        img.save_decod_img(torch.divide(perturbed_images, 255), str(self.i), cfg)
        return perturbed_images

