import json
import torch
import torch.nn as nn
import numpy as np
import os
from PIL import Image
from torchvision import models, transforms

# ==============================
# CONFIG
# ==============================

IMAGE_PATH = r"C:\Projects\CV\check\sample.jpg"
MODEL_DIR = r"C:\Projects\CV\models_highacc"
CLASS_NAME = "class_1_fine_texture"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Auto-detect model file
BACKBONE_NAMES = ["convnext_tiny", "convnext_small", "swin_t", "efficientnet_v2_s",
                  "efficientnet_b3", "efficientnetb3", "efficientnetb0"]
MODEL_PATH = None
META_PATH = None
for bb in BACKBONE_NAMES:
    mp = os.path.join(MODEL_DIR, f"{CLASS_NAME}_highacc_{bb}.pth")
    mt = os.path.join(MODEL_DIR, f"{CLASS_NAME}_highacc_{bb}_meta.json")
    if os.path.exists(mp):
        MODEL_PATH = mp
        META_PATH = mt
        break

if MODEL_PATH is None:
    print("ERROR: No model found for", CLASS_NAME)
    raise SystemExit(1)

# ==============================
# LOAD META
# ==============================
threshold = 0.5
image_size = 224
backbone = "efficientnet_b0"
dropout_feat = 0.40
dropout_mid = 0.20
if META_PATH and os.path.exists(META_PATH):
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    threshold = float(meta.get("decision_threshold", 0.5))
    image_size = int(meta.get("image_size", 224))
    backbone = meta.get("backbone", "efficientnet_b0")
    dropout_feat = meta.get("dropout_feat", 0.40)
    dropout_mid = meta.get("dropout_mid", 0.20)

# ==============================
# BUILD MODEL (matches training architecture)
# ==============================
def build_model():
    if backbone == "convnext_tiny":
        model = models.convnext_tiny(weights=None)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Sequential(
            nn.Dropout(p=dropout_feat), nn.Linear(in_features, 128),
            nn.GELU(), nn.Dropout(p=dropout_mid), nn.Linear(128, 1),
        )
    elif backbone == "convnext_small":
        model = models.convnext_small(weights=None)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Sequential(
            nn.Dropout(p=dropout_feat), nn.Linear(in_features, 128),
            nn.GELU(), nn.Dropout(p=dropout_mid), nn.Linear(128, 1),
        )
    elif backbone == "swin_t":
        model = models.swin_t(weights=None)
        in_features = model.head.in_features
        model.head = nn.Sequential(
            nn.Dropout(p=dropout_feat), nn.Linear(in_features, 128),
            nn.GELU(), nn.Dropout(p=dropout_mid), nn.Linear(128, 1),
        )
    elif backbone == "efficientnet_v2_s":
        model = models.efficientnet_v2_s(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_feat, inplace=True), nn.Linear(in_features, 128),
            nn.SiLU(inplace=True), nn.Dropout(p=dropout_mid), nn.Linear(128, 1),
        )
    elif backbone == "efficientnet_b3":
        model = models.efficientnet_b3(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_feat, inplace=True), nn.Linear(in_features, 128),
            nn.SiLU(inplace=True), nn.Dropout(p=dropout_mid), nn.Linear(128, 1),
        )
    elif backbone == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.35, inplace=True), nn.Linear(in_features, 1),
        )
    else:
        raise ValueError(f"Unknown backbone: {backbone}")
    return model

model = build_model()
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
model = model.to(DEVICE)
model.eval()

# ==============================
# PREPROCESSING (matches training eval pipeline)
# ==============================
eval_transform = transforms.Compose([
    transforms.Resize((image_size + 32, image_size + 32)),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ==============================
# INFERENCE WITH 4-VIEW TTA
# ==============================
@torch.no_grad()
def tta_inference(tensor: torch.Tensor) -> float:
    probs = torch.sigmoid(model(tensor)).item()
    probs += torch.sigmoid(model(torch.flip(tensor, dims=[3]))).item()
    probs += torch.sigmoid(model(torch.flip(tensor, dims=[2]))).item()
    probs += torch.sigmoid(model(torch.flip(torch.flip(tensor, dims=[2]), dims=[3]))).item()
    return probs / 4.0

# ==============================
# LOAD IMAGE → PREDICT → PRINT
# ==============================
if not os.path.exists(IMAGE_PATH):
    print(f"ERROR: Image not found: {IMAGE_PATH}")
    raise SystemExit(1)

img = Image.open(IMAGE_PATH).convert("RGB")
tensor = eval_transform(img).unsqueeze(0).to(DEVICE)
prob = tta_inference(tensor)

prediction = "DEFECT" if prob >= threshold else "GOOD"

print("=" * 50)
print(f"  Image      : {IMAGE_PATH}")
print(f"  Backbone   : {backbone}")
print(f"  Image Size : {image_size}")
print(f"  Threshold  : {threshold:.3f}")
print(f"  Probability: {prob:.4f}")
print(f"  Prediction : {prediction}")
print("=" * 50)
