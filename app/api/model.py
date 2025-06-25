import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

class ModelPredictor:
    def __init__(self, model_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        #classes for pneumonia classification
        self.classes = ["NORMAL", "PNEUMONIA"]

        self.model = self._load_model(model_path)

        # Define the image transformations for resnet18 input
        # ResNet18 expects images of size 224x224 with normalization
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def predict(self, pil_image: Image) -> dict:
        #load and preprocess the image
        image = pil_image.convert("RGB")
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        #predict 
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = F.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        return {
            "predicted_class": self.classes[predicted.item()],
            "confidence": float(confidence.item()),
            "all_probabilities": {
                self.classes[i]: float(probabilities[0][i]) 
                for i in range(len(self.classes))
            }
        }
    
    def _load_model(self, model_path: str):
        model = models.resnet18(weights=False)

        num_classes = len(self.classes)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model