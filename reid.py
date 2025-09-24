# reid.py
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models
import numpy as np


class ReIDExtractor:
def __init__(self, device='cpu'):
self.device = device
self.model = models.resnet18(pretrained=True)
# remove final FC
self.model.fc = nn.Identity()
self.model.to(device).eval()
self.transform = T.Compose([
T.ToPILImage(),
T.Resize((128, 256)),
T.ToTensor(),
T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def extract(self, crop_bgr):
# crop_bgr: numpy array BGR (cv2)
import cv2
crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
x = self.transform(crop_rgb).unsqueeze(0).to(self.device)
with torch.no_grad():
feat = self.model(x)
feat = feat.squeeze().cpu().numpy()
# normalize
norm = np.linalg.norm(feat) + 1e-6
feat = feat / norm
return feat