import numpy as np
from model.clip import CLIP


class CLIPModel:
    def __init__(self, config):
        self.config = config.model
        self.device = config.system.device

        if config.algorithm.loss == 'clip':
            init_temp, init_bias = np.log(5), 0
        else:
            init_temp, init_bias = np.log(10), -10

        self.clip = CLIP(
            from_pretrained=self.config.from_pretrained,
            proj_dim=self.config.proj_dim,
            init_temp=init_temp,
            init_bias=init_bias
            ).to(self.device)

    def generate_similarity_matrix(self, image, text, text_mask):
        return self.clip(image, text, text_mask)

    def encode_image(self, image):
        return self.clip.encode_image(image)

    def encode_text(self, text, text_mask):
        return self.clip.encode_text(text, text_mask)