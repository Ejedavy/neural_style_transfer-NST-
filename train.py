import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.optim import Adam
from utils import get_gram, load_image, denormalize
from model import FeatureExtractor
from tqdm import tqdm

# Defining Hyperparameters
EPOCHS = 500
LR = 1e-2
content_weight = 1
style_weight = 50

# Data Preprocessing
device = "cuda" if torch.cuda.is_available() else "cpu"
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean = (0.485,0.456, 0.406), std=(0.229, 0.224, 0.225))])
content = load_image("/Users/davideje/Projects/Neaural_Style_Transfer/images/content/content_1.jpg", transform= transform, device=device)
style = load_image("/Users/davideje/Projects/Neaural_Style_Transfer/images/style/style_1.jpg", transform= transform, device=device)
styled_image = content.clone()
styled_image.requires_grad = True


# Loading the model
extractor = FeatureExtractor(device= device)
for param in extractor.parameters():
    param.requires_grad = False

# Creating the optimizer
optimizer = Adam(lr = LR, params=[styled_image])


for epoch in tqdm(range(EPOCHS)):
    style_loss = 0
    content_feat = extractor(content)
    style_features = extractor(style)
    generated_features = extractor(styled_image)

    content_loss = torch.mean((content_feat[-1] - generated_features[-1])**2)
    for sf, gf in zip(style_features, generated_features):
        _, c, h, w = sf.size()
        gram_style = get_gram(sf)
        gram_generated = get_gram(gf)
        style_loss += torch.mean((gram_style - gram_generated)**2) / (c * h * w)
    loss = (content_weight * content_loss) + (style_loss * style_weight)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Training Epoch: {epoch + 1}/ {EPOCHS}, Content Loss: {content_loss}, Style Loss: {style_loss}, Loss: {loss.item()}")

with torch.no_grad():
    image = styled_image.cpu()
    save_image(image, "/Users/davideje/Projects/Neaural_Style_Transfer/images/generated/generated_1.jpg")
