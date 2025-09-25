# scripts/train.py
import argparse
import yaml
from pathlib import Path
from collections import Counter, defaultdict
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from PIL import Image


# -----------------------------
# Dataset helper
# -----------------------------
class SimpleImageDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples  # list[(path_str, label_int)]
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, y = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), y


# -----------------------------
# Build dataset splits
# -----------------------------
def build_vggface2hq_splits(cfg, train_t, test_t):
    train_root = Path(cfg["train_dir"])
    test_root = Path(cfg["test_dir"])
    val_ratio = float(cfg.get("val_split", 0.15))
    seed = int(cfg.get("seed", 42))

    # 1) Train completo
    ds_train_full = ImageFolder(str(train_root))
    class_names = ds_train_full.classes
    targets = [y for _, y in ds_train_full.samples]
    counts = Counter(targets)

    # 2) Selección top-K con mínimo de imágenes
    min_per_id = int(cfg.get("min_images_per_id", 1))
    top_k = int(cfg.get("top_k_identities", len(class_names)))
    eligible = [c for c, n in counts.most_common() if n >= min_per_id]
    sel_old_idxs = eligible[:top_k]
    old2new = {old: i for i, old in enumerate(sel_old_idxs)}
    new_class_names = [class_names[old] for old in sel_old_idxs]

    # 3) Agrupar muestras por clase seleccionada
    per_class = defaultdict(list)
    for p, old in ds_train_full.samples:
        if old in old2new:
            per_class[old2new[old]].append(p)

    # 4) Split estratificado por clase
    train_samples, val_samples = [], []
    random.seed(seed)
    for new_lbl, paths in per_class.items():
        random.shuffle(paths)
        n = len(paths)
        n_val = max(1, int(n * val_ratio))
        val_paths = paths[:n_val]
        train_paths = paths[n_val:]
        val_samples += [(p, new_lbl) for p in val_paths]
        train_samples += [(p, new_lbl) for p in train_paths]

    # 5) Test: filtrar a clases seleccionadas
    ds_test_full = ImageFolder(str(test_root))
    test_samples = []
    for p, old in ds_test_full.samples:
        cls_name = ds_test_full.classes[old]
        if cls_name in new_class_names:
            new_lbl = new_class_names.index(cls_name)
            test_samples.append((p, new_lbl))

    # 6) Datasets finales
    ds_train = SimpleImageDataset(train_samples, transform=train_t)
    ds_val = SimpleImageDataset(val_samples, transform=test_t)
    ds_test = SimpleImageDataset(test_samples, transform=test_t)

    # 7) Mapeos
    label2id = {i: ident for i, ident in enumerate(new_class_names)}
    id2label = {ident: i for i, ident in label2id.items()}

    return ds_train, ds_val, ds_test, label2id, id2label


# -----------------------------
# DataLoaders
# -----------------------------
def get_dataloaders(cfg):
    img_size = int(cfg.get("img_size", 224))
    train_t = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    test_t = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    ds_train, ds_val, ds_test, label2id, id2label = build_vggface2hq_splits(cfg, train_t, test_t)

    dl_train = DataLoader(ds_train, batch_size=cfg["batch_size"], shuffle=True,
                          num_workers=cfg.get("num_workers", 4), pin_memory=True)
    dl_val = DataLoader(ds_val, batch_size=cfg["batch_size"], shuffle=False,
                        num_workers=cfg.get("num_workers", 4), pin_memory=True)
    dl_test = DataLoader(ds_test, batch_size=cfg["batch_size"], shuffle=False,
                         num_workers=cfg.get("num_workers", 4), pin_memory=True)

    return ds_train, ds_val, ds_test, dl_train, dl_val, dl_test, label2id, id2label


# -----------------------------
# Train loop
# -----------------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        pred = out.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            total_loss += loss.item() * x.size(0)
            pred = out.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return total_loss / total, correct / total


# -----------------------------
# Main
# -----------------------------
def main(cfg):

    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    # Fuerza tipos por si el YAML trae strings
    epochs       = int(cfg.get("epochs", 10))
    batch_size   = int(cfg.get("batch_size", 64))
    num_workers  = int(cfg.get("num_workers", 4))
    lr           = float(cfg.get("lr", 1e-4))
    weight_decay = float(cfg.get("weight_decay", 1e-4))

    # Vuelve a construir los dataloaders usando los ints (opcional si ya lo haces dentro)
    ds_train, ds_val, ds_test, dl_train, dl_val, dl_test, label2id, id2label = get_dataloaders({
        **cfg,
        "batch_size": batch_size,
        "num_workers": num_workers,
    })
    num_classes = len(label2id)

    # Modelo (API nueva sin 'pretrained')
    weights = models.ResNet50_Weights.IMAGENET1K_V1
    model = models.resnet50(weights=weights)
    in_feats = model.fc.in_features
    model.fc = nn.Linear(in_feats, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_acc = 0.0
    out_path = Path(cfg["model_out"]); out_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        tr_loss, tr_acc = train_one_epoch(model, dl_train, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, dl_val, criterion, device)
        print(f"[Epoch {epoch+1}] Train loss={tr_loss:.4f}, acc={tr_acc:.4f} | Val loss={val_loss:.4f}, acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "state_dict": model.state_dict(),
                "img_size": cfg.get("img_size", 224),
                "label2id": label2id,
                "id2label": id2label,
            }, out_path)
            print(f"  >> Mejor modelo guardado en {out_path} (val_acc={val_acc:.4f})")


# -----------------------------
# Entry point
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    main(cfg)
