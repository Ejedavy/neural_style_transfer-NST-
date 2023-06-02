import torch
import torch.nn as nn
from torchvision.models import vgg16

class FeatureExtractor(nn.Module):
    def __init__(self, device):
        super(FeatureExtractor, self).__init__()
        self.selected_layers = [3,6,8,11,13,15,18,20,22,25,27,29]
        self.model = vgg16(pretrained = True).features

    def forward(self, x):
        layer_features = []
        for layer_id, layer in self.model._modules.items():
            x = layer(x)
            if int(layer_id) in self.selected_layers:
                layer_features.append(x)
        return layer_features 

