from torch import nn
from torchvision import models
from torchvision.models import ResNet50_Weights


class ImageEncoder(nn.Module):
    def __init__(self, from_pretrained, proj_dim):
        super(ImageEncoder, self).__init__()
        if from_pretrained:
            self.resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        else:
            self.resnet = models.resnet50()

        # remove fc layer
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        output_dim = 2048
        self.projection = nn.Sequential(
            nn.Linear(output_dim, output_dim, bias=False),
            nn.ReLU(),
            nn.Linear(output_dim, proj_dim, bias=False))

    def forward(self, image):
        output = self.resnet(image)
        output = output.view(output.size(0), -1) # batch_size, hidden_dim
        projection = self.projection(output)
        return projection