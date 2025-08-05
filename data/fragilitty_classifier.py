import os
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from PIL import Image

# --- Configuration ---
image_dir = "data/training/items"
batch_size = 4
num_epochs = 5
learning_rate = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Image Transform ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# --- Load Dataset ---
dataset = ImageFolder(root=image_dir, transform=transform)
label_map = {v: k for k, v in dataset.class_to_idx.items()}
print("Class labels:", label_map)

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# --- Define Model ---
model = resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)  # Binary classification
model = model.to(device)

# --- Training Setup ---
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# --- Training Loop ---
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    acc = correct / total
    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {running_loss:.4f} - Accuracy: {acc:.2%}")

# --- Save Model ---
model_path = "models/fragility_classifier.pth"
os.makedirs(os.path.dirname(model_path), exist_ok=True)
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")

# --- Optional: Prediction Function ---
def predict_image(image_path):
    model.eval()
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    output = model(image)
    _, predicted = torch.max(output, 1)
    return label_map[int(predicted.item())]

# Example usage (uncomment to test)
# print(predict_image("c:/WareFlowAI/sample.jpg"))