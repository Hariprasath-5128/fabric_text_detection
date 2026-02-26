import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


DEFAULT_SPLIT_ROOT = Path(r"C:\Projects\CV\tilda_structured_split")
DEFAULT_OUT_DIR = Path(r"C:\Projects\CV\models_highacc")


class FabricBinaryDataset(Dataset):
    def __init__(self, samples: List[Tuple[str, int]], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, torch.tensor([label], dtype=torch.float32)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, thr: float) -> Dict[str, float]:
    y_pred = (y_prob >= thr).astype(np.int32)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    total = len(y_true)
    acc = (tp + tn) / total if total else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    bal_acc = ((tp / max(1, tp + fn)) + (tn / max(1, tn + fp))) / 2.0
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
            "balanced_accuracy": bal_acc, "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def collect_samples(class_dir: Path, split_name: str) -> List[Tuple[str, int]]:
    out: List[Tuple[str, int]] = []
    for ext in ("*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg", "*.bmp"):
        for p in (class_dir / split_name / "no_defect").rglob(ext):
            out.append((str(p), 0))
        for p in (class_dir / split_name / "defect").rglob(ext):
            out.append((str(p), 1))
    return out


def discover_classes(split_root: Path) -> List[str]:
    return sorted([d.name for d in split_root.iterdir() if d.is_dir() and d.name.startswith("class_")])


def build_model_from_meta(meta: Dict) -> nn.Module:
    """Build model architecture exactly matching what was trained, using meta.json."""
    backbone = meta.get("backbone", "efficientnet_b0")
    dropout_feat = meta.get("dropout_feat", 0.40)
    dropout_mid = meta.get("dropout_mid", 0.20)

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
        raise ValueError(f"Unknown backbone in meta: {backbone}")

    return model


def find_model_files(out_dir: Path, class_name: str) -> Tuple[Path, Path]:
    """Find model + meta files for a class, trying all known backbone patterns."""
    backbones = ["convnext_tiny", "convnext_small", "swin_t", "efficientnet_v2_s",
                 "efficientnet_b3", "efficientnetb3", "efficientnetb0"]
    for bb in backbones:
        model_path = out_dir / f"{class_name}_highacc_{bb}.pth"
        meta_path = out_dir / f"{class_name}_highacc_{bb}_meta.json"
        if model_path.exists():
            return model_path, meta_path
    return Path(""), Path("")


@torch.no_grad()
def evaluate_one_class(
    class_name: str,
    split_root: Path,
    out_dir: Path,
    batch_size: int,
    device: torch.device,
) -> Dict[str, float]:
    class_dir = split_root / class_name
    test_samples = collect_samples(class_dir, "test")
    if not test_samples:
        test_samples = collect_samples(class_dir, "split")
    if not test_samples:
        return {"class_name": class_name, "error": "missing_test_or_split"}

    model_path, meta_path = find_model_files(out_dir, class_name)
    if not model_path.exists():
        return {"class_name": class_name, "error": f"no_model_found_for_{class_name}"}

    # Load meta
    meta = {"backbone": "efficientnet_b0", "decision_threshold": 0.5, "image_size": 224}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

    threshold = float(meta.get("decision_threshold", 0.5))
    image_size = int(meta.get("image_size", 224))
    backbone = meta.get("backbone", "efficientnet_b0")

    eval_tf = transforms.Compose([
        transforms.Resize((image_size + 32, image_size + 32)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    ds = FabricBinaryDataset(test_samples, transform=eval_tf)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    model = build_model_from_meta(meta).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    crop_size = image_size
    probs_all: List[np.ndarray] = []
    y_all: List[np.ndarray] = []
    for x, y in dl:
        x = x.to(device)
        batch_probs = torch.zeros(x.size(0), device=device)

        views = [
            x,
            torch.flip(x, dims=[3]),
            torch.flip(x, dims=[2]),
            torch.flip(torch.flip(x, dims=[2]), dims=[3]),
        ]
        pad = 8
        x_padded = torch.nn.functional.pad(x, [pad, pad, pad, pad], mode='reflect')
        views.append(x_padded[:, :, :crop_size, :crop_size])
        views.append(x_padded[:, :, 2*pad:2*pad+crop_size, 2*pad:2*pad+crop_size])
        views.append(x_padded[:, :, pad:pad+crop_size, :crop_size])
        views.append(x_padded[:, :, pad:pad+crop_size, 2*pad:2*pad+crop_size])

        n_views = len(views)
        for v in views:
            logits = model(v)
            batch_probs += torch.sigmoid(logits).squeeze(1)
        batch_probs /= n_views

        probs_all.append(batch_probs.cpu().numpy())
        y_all.append(y.numpy().reshape(-1).astype(np.int32))

    probs = np.concatenate(probs_all) if probs_all else np.array([])
    y_true = np.concatenate(y_all) if y_all else np.array([])
    m = compute_metrics(y_true, probs, threshold)
    return {
        "class_name": class_name,
        "backbone": backbone,
        "threshold": float(threshold),
        "accuracy": float(m["accuracy"]),
        "precision": float(m["precision"]),
        "recall": float(m["recall"]),
        "f1": float(m["f1"]),
        "balanced_accuracy": float(m["balanced_accuracy"]),
        "tp": int(m["tp"]), "tn": int(m["tn"]),
        "fp": int(m["fp"]), "fn": int(m["fn"]),
        "total": int(len(test_samples)),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate per-class models (auto-detects backbone from meta).")
    p.add_argument("--split-root", type=str, default=str(DEFAULT_SPLIT_ROOT))
    p.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR))
    p.add_argument("--batch-size", type=int, default=8)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    split_root = Path(args.split_root)
    out_dir = Path(args.out_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    classes = discover_classes(split_root)
    if not classes:
        raise RuntimeError(f"No class_* directories found in {split_root}")

    print(f"DEVICE={device}")
    print("\nCLASSWISE_RESULTS")

    results: List[Dict[str, float]] = []
    g_tp = g_tn = g_fp = g_fn = g_total = 0
    for class_name in classes:
        r = evaluate_one_class(class_name, split_root, out_dir, args.batch_size, device)
        results.append(r)
        if "error" in r:
            print(f"{class_name} | error={r['error']}")
            continue
        print(
            f"{class_name} ({r.get('backbone','?')}) | thr={r['threshold']:.3f} | "
            f"acc={r['accuracy']:.4f} bal_acc={r['balanced_accuracy']:.4f} | "
            f"prec={r['precision']:.4f} rec={r['recall']:.4f} f1={r['f1']:.4f} | "
            f"tp={r['tp']} tn={r['tn']} fp={r['fp']} fn={r['fn']} total={r['total']}"
        )
        g_tp += r["tp"]
        g_tn += r["tn"]
        g_fp += r["fp"]
        g_fn += r["fn"]
        g_total += r["total"]

    if g_total > 0:
        acc = (g_tp + g_tn) / g_total
        prec = g_tp / (g_tp + g_fp) if (g_tp + g_fp) else 0.0
        rec = g_tp / (g_tp + g_fn) if (g_tp + g_fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        print(
            f"\nOVERALL_RESULTS | acc={acc:.4f} | prec={prec:.4f} | rec={rec:.4f} | f1={f1:.4f} | "
            f"tp={g_tp} tn={g_tn} fp={g_fp} fn={g_fn} total={g_total}"
        )
    else:
        print("\nOVERALL_RESULTS | no evaluated samples")

    (out_dir / "evaluation_summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
