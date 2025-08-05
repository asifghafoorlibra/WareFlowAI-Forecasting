import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

# --- Configuration ---
MODEL_PATH = "models/fragility_classifier_new.pth"
IMAGE_PATH = "data/product_images/man.jpg"
IMG_SIZE = (224, 224)
THRESHOLD = 0.4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABEL_MAP = {0: "Fragile", 1: "Non-Fragile"}
DEBUG = False  # Set to True to print raw logits and probabilities

# --- Preprocessing ---
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor()
    ])
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(DEVICE)
    return tensor, image

# --- Load Model ---
def load_model():
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

# --- Grad-CAM ---
def generate_gradcam(model, input_tensor, target_class):
    gradients = []
    activations = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    target_layer = model.layer4[-1].conv2
    target_layer.register_forward_hook(forward_hook)
    target_layer.register_backward_hook(backward_hook)

    output = model(input_tensor)
    loss = output[:, target_class]
    model.zero_grad()
    loss.backward()

    grad = gradients[0].cpu().numpy()[0]
    act = activations[0].cpu().numpy()[0]

    weights = np.mean(grad, axis=(1, 2))
    cam = np.zeros(act.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * act[i]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, IMG_SIZE)
    cam = cam / cam.max()
    return cam

# --- Visualization ---
def show_gradcam(original_image, heatmap, alpha=0.4):
    img = np.array(original_image)
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    superimposed = heatmap_color * alpha + img
    plt.imshow(superimposed.astype(np.uint8))
    plt.axis('off')
    plt.title("Grad-CAM")
    plt.show()

# --- Prediction ---
def predict_fragility(image_path):
    model = load_model()
    input_tensor, original_image = preprocess_image(image_path)

    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output, dim=1)
        pred_index = probs.argmax(dim=1).item()
        confidence = round(probs[0][pred_index].item() * 100, 2)
        label = LABEL_MAP[pred_index]

    print(f"Prediction: {label} ({confidence}% confidence)")

    if DEBUG:
        print("Raw logits:", output.cpu().numpy())
        print("Softmax probabilities:", probs.cpu().numpy())
        print("Predicted class index:", pred_index)

    heatmap = generate_gradcam(model, input_tensor, target_class=pred_index)
    show_gradcam(original_image, heatmap)

# --- Run ---
if __name__ == "__main__":
    predict_fragility(IMAGE_PATH)