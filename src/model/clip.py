import torch
import torch.nn as nn


class CLIP(nn.Module):
    def __init__(self):
        self.image_encoder = None
        self.image_projection = None

        self.text_encoder = None
        self.text_projection = None

    def forward(self, image, text, text_mask):
        pass

    def encode_image(self, image):
        pass

    def encode_text(self, text):
        pass

