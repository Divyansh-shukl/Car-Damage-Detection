import os
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image

trained_model = None
class_names = ['Front Breakage', 'Front Crushed', 'Front Normal', 'Rear Breakage', 'Rear Crushed', 'Rear Normal']

class CarClassifierResNet(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        # If you already have pretrained weights available locally, you can set weights=None to avoid downloads.
        # Using weights='DEFAULT' may try to download weights if not present.
        self.model = models.resnet50(weights='DEFAULT')
        # Freeze all layers except layer4 and fc
        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.layer4.parameters():
            param.requires_grad = True

        # Replace final fc
        self.model.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.model.fc.in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)


def predict(image_path):
    # image transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)  # shape: (1, 3, 224, 224)

    global trained_model

    if trained_model is None:
        # build absolute model path relative to this file
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, "model", "saved_model.pth")

        print("DEBUG: Loading model from:", model_path)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # pick device safely
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # load state dict mapped to device to avoid CUDA<->CPU mismatch errors
        # If your torch supports weights_only and you want extra safety, you can try weights_only=True
        state_dict = torch.load(model_path, map_location=device)

        trained_model = CarClassifierResNet()
        trained_model.load_state_dict(state_dict)
        trained_model.to(device)
        trained_model.eval()

    # ensure tensor is on same device as model
    device = next(trained_model.parameters()).device
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        output = trained_model(image_tensor)
        _, predicted_class = torch.max(output, 1)
        return class_names[predicted_class.item()]
