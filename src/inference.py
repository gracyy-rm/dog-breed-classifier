import os
import torch
import pickle
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet50

# ---------------- Device ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- Base Path (IMPORTANT) ----------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------- Paths ----------------
model_path = os.path.join(BASE_DIR, "outputs", "models", "dog_breed_model.pth")
label_path = os.path.join(BASE_DIR, "outputs", "models", "label_encoder.pkl")

# ---------------- Model ----------------
model = resnet50(weights=None)
model.fc = torch.nn.Linear(2048, 120)  # 120 classes

model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# ---------------- Label Encoder ----------------
with open(label_path, "rb") as f:
    le = pickle.load(f)

# ---------------- Transform ----------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ---------------- Predict (Single) ----------------
def predict(image_file):
    image = Image.open(image_file).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        _, pred = torch.max(outputs, 1)

    return le.inverse_transform([pred.item()])[0]


# ---------------- Predict Top-K (Optional Advanced) ----------------
def predict_topk(image_file, k=3):
    image = Image.open(image_file).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.nn.functional.softmax(outputs, dim=1)

        topk_probs, topk_indices = torch.topk(probs, k)

    results = []
    for i in range(k):
        label = le.inverse_transform([topk_indices[0][i].item()])[0]
        confidence = topk_probs[0][i].item() * 100
        results.append((label, confidence))

    return results
