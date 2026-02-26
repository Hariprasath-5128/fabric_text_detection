import cv2
import time
import json
import torch
import torch.nn as nn
import numpy as np
import os
from torchvision import models

# ==============================
# CONFIG
# ==============================
STREAM_URL = "http://10.159.194.193:8080/video"

# Search for model files — try per-class backbone names, then legacy B3/B0
MODEL_DIR = r"C:\Projects\CV\models_highacc"
CLASS_NAME = "class_1_fine_texture"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_BASE = r"C:\Projects\CV\saved_images"

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
        print(f"Found model: {bb}")
        break

if MODEL_PATH is None:
    print("ERROR: No model found for", CLASS_NAME)
    raise SystemExit(1)

# Create folder structure
for cls in ["good", "defect"]:
    os.makedirs(os.path.join(SAVE_BASE, cls, "original"), exist_ok=True)
    os.makedirs(os.path.join(SAVE_BASE, cls, "processed"), exist_ok=True)

# ==============================
# LOAD META
# ==============================
threshold = 0.5
image_size = 288
backbone = "convnext_tiny"
dropout_feat = 0.45
dropout_mid = 0.20
if META_PATH and os.path.exists(META_PATH):
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    threshold = float(meta.get("decision_threshold", 0.5))
    image_size = int(meta.get("image_size", 288))
    backbone = meta.get("backbone", "convnext_tiny")
    dropout_feat = meta.get("dropout_feat", 0.45)
    dropout_mid = meta.get("dropout_mid", 0.20)

print(f"Using threshold={threshold:.3f}, image_size={image_size}, backbone={backbone}")

# ==============================
# MODEL ARCHITECTURE (matches training)
# ==============================
def build_model():
    if backbone == "convnext_tiny":
        model = models.convnext_tiny(weights=None)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Sequential(
            nn.Dropout(p=dropout_feat),
            nn.Linear(in_features, 128),
            nn.GELU(),
            nn.Dropout(p=dropout_mid),
            nn.Linear(128, 1),
        )
    elif backbone == "convnext_small":
        model = models.convnext_small(weights=None)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Sequential(
            nn.Dropout(p=dropout_feat),
            nn.Linear(in_features, 128),
            nn.GELU(),
            nn.Dropout(p=dropout_mid),
            nn.Linear(128, 1),
        )
    elif backbone == "swin_t":
        model = models.swin_t(weights=None)
        in_features = model.head.in_features
        model.head = nn.Sequential(
            nn.Dropout(p=dropout_feat),
            nn.Linear(in_features, 128),
            nn.GELU(),
            nn.Dropout(p=dropout_mid),
            nn.Linear(128, 1),
        )
    elif backbone == "efficientnet_v2_s":
        model = models.efficientnet_v2_s(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_feat, inplace=True),
            nn.Linear(in_features, 128),
            nn.SiLU(inplace=True),
            nn.Dropout(p=dropout_mid),
            nn.Linear(128, 1),
        )
    elif backbone == "efficientnet_b3":
        model = models.efficientnet_b3(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_feat, inplace=True),
            nn.Linear(in_features, 128),
            nn.SiLU(inplace=True),
            nn.Dropout(p=dropout_mid),
            nn.Linear(128, 1),
        )
    elif backbone == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.35, inplace=True),
            nn.Linear(in_features, 1),
        )
    else:
        raise ValueError(f"Unknown backbone: {backbone}")
    return model

print("Loading model...")
model = build_model()
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
model = model.to(DEVICE)
model.eval()
print("Model loaded successfully")

# ImageNet normalization
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

def preprocess_bgr(frame_bgr: np.ndarray, size: int) -> tuple:
    resized = cv2.resize(frame_bgr, (size + 32, size + 32), interpolation=cv2.INTER_AREA)
    y0 = (resized.shape[0] - size) // 2
    x0 = (resized.shape[1] - size) // 2
    crop = resized[y0:y0 + size, x0:x0 + size]
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - MEAN) / STD
    chw = np.transpose(rgb, (2, 0, 1))
    tensor = torch.from_numpy(chw).unsqueeze(0).to(DEVICE)
    return tensor, crop

def tta_inference(tensor: torch.Tensor) -> float:
    """4-view TTA for real-time inference."""
    with torch.no_grad():
        probs = torch.sigmoid(model(tensor)).item()
        probs += torch.sigmoid(model(torch.flip(tensor, dims=[3]))).item()
        probs += torch.sigmoid(model(torch.flip(tensor, dims=[2]))).item()
        probs += torch.sigmoid(model(torch.flip(torch.flip(tensor, dims=[2]), dims=[3]))).item()
    return probs / 4.0

# ==============================
# STREAM
# ==============================
print("Connecting to mobile stream...")
cap = cv2.VideoCapture(STREAM_URL)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("Cannot connect to stream")
    raise SystemExit(1)

print("Live stream started (Press Q to quit)")
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Stream error")
        break

    input_tensor, processed_bgr = preprocess_bgr(frame, image_size)
    prob = tta_inference(input_tensor)

    prediction = 1 if prob >= threshold else 0
    cls_folder = "defect" if prediction == 1 else "good"
    label_text = f"{cls_folder.upper()} ({prob:.2f})"

    color = (0, 0, 255) if prediction == 1 else (0, 255, 0)
    cv2.putText(frame, label_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.imshow(f"Live Defect Detection - {CLASS_NAME}", frame)

    if frame_count % 30 == 0:
        timestamp = int(time.time() * 1000)
        original_path = os.path.join(SAVE_BASE, cls_folder, "original", f"{cls_folder}_{timestamp}.jpg")
        processed_path = os.path.join(SAVE_BASE, cls_folder, "processed", f"{cls_folder}_{timestamp}.jpg")
        cv2.imwrite(original_path, frame)
        cv2.imwrite(processed_path, processed_bgr)
        print("-" * 50)
        print(f"Frame Index    : {frame_count}")
        print(f"Prediction     : {cls_folder.upper()}")
        print(f"Probability    : {prob:.4f}")
        print(f"Original Saved : {original_path}")
        print(f"Processed Saved: {processed_path}")
        print("-" * 50)

    frame_count += 1
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
