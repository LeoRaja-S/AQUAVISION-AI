import torch
from torchvision import models, transforms
from PIL import Image

# Classes
class_names = [
    'Bacterial diseases - Aeromoniasis',
    'Bacterial gill disease',
    'Bacterial Red disease',
    'Fungal diseases Saprolegniasis',
    'Healthy Fish',
    'Parasitic diseases',
    'Viral diseases White tail disease'
]

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
cnn_model = models.resnet18(pretrained=False)
cnn_model.fc = torch.nn.Linear(cnn_model.fc.in_features, len(class_names))
cnn_model.load_state_dict(torch.load("models/fish_cnn.pth", map_location=device))
cnn_model.to(device)
cnn_model.eval()

# Image transform
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
])

# Prediction
def classify_health(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = cnn_model(image)
        _, preds = torch.max(outputs, 1)
    return class_names[preds.item()]
