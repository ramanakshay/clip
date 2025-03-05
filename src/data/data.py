import torch
from torchvision import datasets
import torchvision.transforms as transforms
from transformers import DistilBertTokenizer
from torch.utils.data import DataLoader
import random

class ImageTransform:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize((224,224)), # resnet input
            transforms.ToTensor()
        ])

    def __call__(self, image):
        return self.transform(image)


class CaptionTransform:
    def __init__(self, tokenizer, context_length, shuffle_captions):
        self.context_length = context_length
        self.shuffle_captions = shuffle_captions
        self.tokenizer = tokenizer

    def __call__(self, captions):
        if self.shuffle_captions:
            caption = random.choice(captions)
        else:
            caption = captions[0]

        instance = self.tokenizer(caption,
                              padding="max_length",
                              truncation=True,
                              max_length=self.context_length)

        tokenized_caption = {key:torch.tensor(value) for key,value in instance.items()}
        return tokenized_caption


class COCOCaptionsData:
    def __init__(self, config):
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        self.config = config.data
        
        self.dataset = datasets.CocoCaptions(
            root=self.config.images_path,
            annFile=self.config.annotations_path,
            transform=ImageTransform(),
            target_transform=CaptionTransform(self.tokenizer,
                                                self.config.context_length,
                                                self.config.shuffle_captions)
            )

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            num_workers=4,
            drop_last=True,
            shuffle=True
            )

    def get_dataloader(self):
        return self.dataloader

