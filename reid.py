import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2

class ReID:
    def __init__(self, model_name="resnet18"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = getattr(models, model_name)(pretrained=True).to(self.device)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def extract_embedding(self, crop):
        """Extract appearance embedding from a person crop."""
        if crop is None or crop.size == 0:
            return np.zeros(512, dtype=np.float32)
        img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        img = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.model(img).cpu().numpy().flatten()
        return embedding / np.linalg.norm(embedding)
