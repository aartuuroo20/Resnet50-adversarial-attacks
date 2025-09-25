import torch
import yaml

from torchvision import models, transforms
from PIL import Image
from pathlib import Path
from attacks import fgsm, pgd, predict_label  

def denormalize_to_01(x_norm: torch.Tensor):
    mean = torch.tensor([0.485, 0.456, 0.406], device=x_norm.device).view(1,3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=x_norm.device).view(1,3,1,1)
    x = x_norm * std + mean
    return x.clamp(0, 1)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = yaml.safe_load(open("config.yaml", "r"))
    ckpt = torch.load(cfg["model_out"], map_location=device)
    num_classes = len(ckpt["id2label"])

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(ckpt["state_dict"])
    model.eval().to(device)

    tf = transforms.Compose([
        transforms.Resize((ckpt.get("img_size", 224), ckpt.get("img_size", 224))),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    img_path = Path(cfg["data_root"]) / "Img" / "img_align_celeba" / "000001.jpg"
    if not img_path.exists():
        raise FileNotFoundError(f"No existe la imagen: {img_path}")

    img = Image.open(img_path).convert("RGB")
    x = tf(img).unsqueeze(0).to(device)

    y_pred = predict_label(model, x)
    print("Pred original (label):", int(y_pred))

    eps_px = 8  
    x_fgsm = fgsm(model, x, y_pred, eps_px=eps_px, targeted=False)
    y_fgsm = predict_label(model, x_fgsm)
    print(f"FGSM (eps={eps_px}px) -> pred:", int(y_fgsm))

    x_pgd = pgd(model, x, y_pred, eps_px=8, alpha_px=2, steps=10, rand_init=True, targeted=False)
    y_pgd = predict_label(model, x_pgd)
    print("PGD (eps=8px, alpha=2px, steps=10) -> pred:", int(y_pgd))

    outdir = Path("attacks_out"); outdir.mkdir(exist_ok=True)
    to_save = [
        ("original.png", denormalize_to_01(x)),
        ("fgsm.png",     denormalize_to_01(x_fgsm)),
        ("pgd.png",      denormalize_to_01(x_pgd)),
    ]
    for name, tensor in to_save:
        from torchvision.utils import save_image
        save_image(tensor, outdir / name)
    print(f"Imágenes guardadas en {outdir.resolve()}")

if __name__ == "__main__":
    main()
