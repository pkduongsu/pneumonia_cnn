import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import accuracy_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

#using ResNet18 as the base model

class PneumoniaDataset(Dataset):

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        for label in ['NORMAL', 'PNEUMONIA']:
            class_dir = os.path.join(root_dir, label)
            print(f"Loading images from: {class_dir}")
            print(f"Number of images in {label}: {len(os.listdir(class_dir))}")
            for img_name in os.listdir(class_dir):
                self.image_paths.append(os.path.join(class_dir, img_name))
                self.labels.append(0 if label == 'NORMAL' else 1) #0 for normal, 1 for Pneumonia

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label 
    
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = PneumoniaDataset(root_dir='data/train', transform=transform)
val_dataset = PneumoniaDataset(root_dir='data/val', transform=transform)
test_dataset = PneumoniaDataset(root_dir='data/test', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, 2) #2 output neurons for NORMAL, PNEUMONIA
model = model.to(device) #send model to GPU

criterion = nn.CrossEntropyLoss() #loss function
optimizer = optim.Adam(model.parameters(), lr=0.001) #optimizer, fine-tuning the whole model

num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0 #reset for each epoch

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad() #zero the gradients
        outputs = model(images) #forward pass
        loss = criterion(outputs, labels)
        loss.backward() #backward pass

        optimizer.step() #update weights

        running_loss += loss

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}")

    model.eval() 
    val_labels = []
    val_preds = []

    with torch.no_grad(): #just evaluating, no gradients needed
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1) #get the top output 

            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

    val_accuracy = accuracy_score(val_labels, val_preds)
    print(f"Validation Accuracy: {val_accuracy}")

model.eval()
test_labels = []
test_preds = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        test_preds.extend(preds.cpu().numpy())
        test_labels.extend(labels.cpu().numpy())

test_accuracy = accuracy_score(test_labels, test_preds)
print(f"Test Accuracy: {test_accuracy}")

#save the model locally
torch.save(model.state_dict(), 'pneumonia_detection_model.pth')