import torch
import torch.nn as nn
import torch.nn.functional as F
from model.clip.encoders import TextEncoder, ImageEncoder


class CLIP(nn.Module):
    def __init__(self, from_pretrained, proj_dim, init_temp=None, init_bias=None):
        super(CLIP, self).__init__()
        self.image_encoder = ImageEncoder(from_pretrained, proj_dim)
        self.text_encoder = TextEncoder(from_pretrained, proj_dim)
        self.temp = nn.Parameter(torch.ones([]) * init_temp)
        self.bias = nn.Parameter(torch.ones([]) * init_bias)

    def forward(self, image, text, text_mask):
        image_encodings = self.encode_image(image)
        text_encodings = self.encode_text(text, text_mask)
        # normalize features
        image_encodings = F.normalize(image_encodings, p=2, dim=-1)
        text_encodings = F.normalize(text_encodings, p=2, dim=-1)
        return image_encodings @ text_encodings.t() * self.temp.exp() + self.bias

    def encode_image(self, image):
        return self.image_encoder(image)

    def encode_text(self, text, text_mask):
        return self.text_encoder(text, text_mask)

