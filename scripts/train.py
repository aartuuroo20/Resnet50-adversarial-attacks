import argparse, os, random, yaml, math
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from pathlib import Path
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms, models
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split

class CelebAIdentities(Dataset):
    """
    Dataset de identidades para CelebA usando los metadatos:
    - identity_CelebA.txt: mapping filename -> identity_id (entero)
    - list_eval_partition.txt: split original (0 train, 1 val, 2 test) (no lo usamos estrictamente)
    Trabaja sobre el directorio 'img_align_celeba/img_align_celeba'.
    Permite filtrar top-K identidades por frecuencia y mínimo de imágenes por identidad.
    """
    def __init__(self, root, split='train', transform=None,
                 top_k=None, min_images=0, seed=42, download=False):
        self.root = Path(root)
        self.transform = transform
        self.split = split
        self.seed = seed
        # Descarga con torchvision para asegurar estructura y anotaciones
        
        if download:
        # No pedimos 'attr' para que no exija list_attr_celeba.txt
            _ = datasets.CelebA(root=str(self.root.parent), split='train',
                        target_type=[], download=True)

        anno_dir = self.root / "Anno"
        img_dir = self.root / "Img" / "img_align_celeba"
        nested = img_dir / "img_align_celeba"  # por si hay carpeta duplicada
        if nested.exists():
            img_dir = nested
        if not img_dir.exists():
            raise FileNotFoundError(f"No encuentro carpeta de imágenes en {img_dir}")

        # Cargar tablas
        ids = pd.read_csv(anno_dir / "identity_CelebA.txt", sep=r"\s+", header=None, names=["filename", "identity"])
        part = pd.read_csv(anno_dir / "list_eval_partition.txt", sep=r"\s+", header=None, names=["filename", "split"])
        df = ids.merge(part, on="filename")

        # Filtrar a solo archivos que existen realmente
        existing = {p.name for p in img_dir.glob("*.jpg")}
        missing_before = df.shape[0]
        df = df[df["filename"].isin(existing)].copy()
        missing_after = missing_before - df.shape[0]
        if missing_after > 0:
            print(f"[Aviso] {missing_after} anotaciones apuntaban a imágenes inexistentes y se han descartado.")

        # Mapear split humano
        split_map = {0: "train", 1: "valid", 2: "test"}
        df["split"] = df["split"].map(split_map)

        # Filtrado por top_k y min_images (se calcula sobre train para simular selección práctica)
        if top_k is not None:
            train_counts = (df[df["split"] == "train"]["identity"]
                            .value_counts())
            keep_ids = train_counts[train_counts >= min_images].head(top_k).index.tolist()
            df = df[df["identity"].isin(keep_ids)]

        # Reindexar labels (0..C-1)
        unique_ids = sorted(df["identity"].unique().tolist())
        self.id2label = {iden: i for i, iden in enumerate(unique_ids)}
        self.label2id = {v: k for k, v in self.id2label.items()}

        # Estratificar a un split propio train/val/test (70/15/15) por si el original no cuadra tras filtrar
        df["label"] = df["identity"].map(self.id2label)
        # Usamos partición estratificada a nivel de archivo tras el filtrado global
        train_df, temp_df = train_test_split(df, test_size=0.30, random_state=seed, stratify=df["label"])
        val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=seed, stratify=temp_df["label"])

        split_map_df = {"train": train_df, "val": val_df, "test": test_df}
        self.df = split_map_df["train" if split=="train" else "val" if split in ["val","valid","validation"] else "test"].reset_index(drop=True)
        self.img_dir = img_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.img_dir / row["filename"]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = int(row["label"])
        return img, label

def get_dataloaders(cfg):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_t = transforms.Compose([
        transforms.Resize((cfg["img_size"], cfg["img_size"])),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
        transforms.ToTensor(), normalize
    ])
    val_t = transforms.Compose([
        transforms.Resize((cfg["img_size"], cfg["img_size"])),
        transforms.ToTensor(), normalize
    ])

    common = dict(root=cfg["data_root"], top_k=cfg["top_k_identities"],
                  min_images=cfg["min_images_per_id"], download=cfg["download"], seed=cfg["seed"])

    ds_train = CelebAIdentities(split='train', transform=train_t, **common)
    ds_val = CelebAIdentities(split='val', transform=val_t, **common)
    ds_test = CelebAIdentities(split='test', transform=val_t, **common)

    train_loader = DataLoader(ds_train, batch_size=cfg["batch_size"], shuffle=True,
                              num_workers=cfg["num_workers"], pin_memory=True)
    val_loader = DataLoader(ds_val, batch_size=cfg["batch_size"], shuffle=False,
                            num_workers=cfg["num_workers"], pin_memory=True)
    test_loader = DataLoader(ds_test, batch_size=cfg["batch_size"], shuffle=False,
                             num_workers=cfg["num_workers"], pin_memory=True)
    return ds_train, ds_val, ds_test, train_loader, val_loader, test_loader

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for x, y in tqdm(loader, desc="Train", leave=False):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * x.size(0)
        preds = logits.argmax(1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return running_loss/total, correct/total

@torch.no_grad()
def evaluate(model, loader, criterion, device, desc="Val"):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    for x, y in tqdm(loader, desc=desc, leave=False):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        running_loss += loss.item() * x.size(0)
        preds = logits.argmax(1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return running_loss/total, correct/total

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(cfg["seed"])
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    device = torch.device(cfg["device"] if torch.cuda.is_available() or cfg["device"]=="cpu" else "cpu")

    ds_train, ds_val, ds_test, dl_train, dl_val, dl_test = get_dataloaders(cfg)
    num_classes = len(ds_train.id2label)

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    # Fine‑tuning completo (puedes congelar si quieres)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    best_val = 0.0
    for epoch in range(1, cfg["epochs"]+1):
        tr_loss, tr_acc = train_one_epoch(model, dl_train, criterion, optimizer, device)
        va_loss, va_acc = evaluate(model, dl_val, criterion, device, desc=f"Val (ep {epoch})")
        print(f"[Epoch {epoch}] Train loss {tr_loss:.4f} acc {tr_acc:.3f} | Val loss {va_loss:.4f} acc {va_acc:.3f}")
        if va_acc > best_val:
            best_val = va_acc
            torch.save({
                "state_dict": model.state_dict(),
                "id2label": ds_train.id2label,
                "label2id": ds_train.label2id,
                "img_size": cfg["img_size"]
            }, cfg["model_out"])
            print(f"✓ Guardado mejor modelo en {cfg['model_out']} (val acc {best_val:.3f})")

    te_loss, te_acc = evaluate(model, dl_test, criterion, device, desc="Test")
    print(f"[Test] loss {te_loss:.4f} acc {te_acc:.3f}")

if __name__ == "__main__":
    main()
