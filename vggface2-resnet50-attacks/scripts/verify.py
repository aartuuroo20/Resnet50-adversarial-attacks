import argparse
import yaml
import torch
import pandas as pd

from PIL import Image
from torchvision import models, transforms
from pathlib import Path


def load_model_and_cfg(cfg_path: str):
    cfg = yaml.safe_load(open(cfg_path, "r"))
    ckpt = torch.load(cfg["model_out"], map_location="cpu")

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = torch.nn.Linear(model.fc.in_features, len(ckpt["id2label"]))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    tf = transforms.Compose([
        transforms.Resize((ckpt.get("img_size", 224), ckpt.get("img_size", 224))),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    return cfg, ckpt, model, tf

def build_filename2label(cfg, ckpt):
    data_root = Path(cfg["data_root"])
    anno = data_root / "Anno" / "identity_CelebA.txt"
    if not anno.exists():
        return {}
    df = pd.read_csv(anno, sep=r"\s+", header=None, names=["filename","identity"])
    id2label = ckpt["id2label"]  # identity_id_original -> label
    df = df[df["identity"].isin(id2label.keys())].copy()
    df["label"] = df["identity"].map(id2label)
    return dict(zip(df["filename"], df["label"]))

def predict(model, tf, img_path: Path):
    img = Image.open(img_path).convert("RGB")
    x = tf(img).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        pred = int(torch.argmax(probs))
    return pred, float(probs[pred]), probs

def topk(probs, k=5):
    vals, idxs = torch.topk(probs, k)
    return [(int(i), float(v)) for v, i in zip(vals, idxs)]

def main():
    ap = argparse.ArgumentParser(description="Verifica si la predicción de una imagen es correcta.")
    ap.add_argument("--config", default="config.yaml", help="Ruta al config.yaml")
    ap.add_argument("--image", required=True, help="Ruta a la imagen (puede ser del dataset o externa)")
    ap.add_argument("--expected", type=int, default=None,
                    help="Etiqueta esperada (label 0..C-1). Útil si la imagen no está en el dataset.")
    ap.add_argument("--topk", type=int, default=5, help="Mostrar Top-K probabilidades")
    args = ap.parse_args()

    cfg, ckpt, model, tf = load_model_and_cfg(args.config)
    filename2label = build_filename2label(cfg, ckpt)
    label2id = ckpt["label2id"]  

    img_path = Path(args.image)
    if not img_path.exists():
        raise FileNotFoundError(f"No existe la imagen: {img_path}")

    pred, conf, probs = predict(model, tf, img_path)
    pred_identity = label2id.get(pred, "?")

    truth_label = None
    if img_path.name in filename2label:
        truth_label = filename2label[img_path.name]
        source = "dataset (identity_CelebA.txt)"
    elif args.expected is not None:
        truth_label = int(args.expected)
        source = "--expected"

    print(f"\nImagen: {img_path.name}")
    print(f"Predicción -> label={pred}  identity_id={pred_identity}  conf={conf:.4f}")

    if truth_label is not None:
        ok = (pred == truth_label)
        truth_identity = label2id.get(truth_label, "?")
        print(f"Ground truth ({source}) -> label={truth_label}  identity_id={truth_identity}")
        print("Resultado:", "CORRECTO" if ok else "INCORRECTO")
        exit_code = 0 if ok else 1
    else:
        print("Ground truth no disponible (ni en dataset ni por --expected).")
        exit_code = 2

    if args.topk > 1:
        print("\nTop-{}:".format(args.topk))
        for rank, (lbl, p) in enumerate(topk(probs, k=args.topk), start=1):
            print(f"  {rank:>2d}. label={lbl:<3d}  identity_id={label2id.get(lbl,'?')}  prob={p:.4f}")

    raise SystemExit(exit_code)

if __name__ == "__main__":
    main()
