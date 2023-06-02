import torch
from PIL import Image
import numpy as np


def load_image(path, transform, size = (300, 300), device= "cpu"):
    image = Image.open(path)
    image = image.resize(size, Image.LANCZOS)
    image = transform(image).unsqueeze(0)
    return image.to(device)

def get_gram(image):
    batch, channels, h, w = image.size()
    image = image.reshape(channels, h * w)
    return torch.mm(image, image.t())

def denormalize(image):
    image = image.numpy().transpose((1, 2, 0))
    mean = np.array([0.485,0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    return (image * std) + mean