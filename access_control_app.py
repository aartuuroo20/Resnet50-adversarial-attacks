import yaml
import streamlit as st
import torch
import torch.nn.functional as F

from pathlib import Path
from PIL import Image
from torchvision import models, transforms

st.set_page_config(page_title="Software de control de acceso", layout="wide")

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)

def to_tensor_and_norm(pil_img, size):
    tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    return tf(pil_img).unsqueeze(0)

def denorm01(x_norm: torch.Tensor) -> torch.Tensor:
    return (x_norm * IMAGENET_STD.to(x_norm.device) + IMAGENET_MEAN.to(x_norm.device)).clamp(0,1)

def to_pil_from01(x01: torch.Tensor) -> Image.Image:
    x = (x01.detach().cpu().squeeze(0).permute(1,2,0).numpy() * 255).astype("uint8")
    return Image.fromarray(x)

def _eps_px_to_norm(eps_px: float, device: torch.device) -> torch.Tensor:
    return (float(eps_px) / 255.0) / IMAGENET_STD.to(device)

def _clamp_norm(x_norm: torch.Tensor) -> torch.Tensor:
    x01 = denorm01(x_norm)
    return (x01 - IMAGENET_MEAN.to(x_norm.device)) / IMAGENET_STD.to(x_norm.device)

def fgsm_targeted(model, x_norm: torch.Tensor, y_target: torch.Tensor, eps_px: float) -> torch.Tensor:
    device = x_norm.device
    x = x_norm.clone().detach().requires_grad_(True)
    logits = model(x)
    loss = F.cross_entropy(logits, y_target)
    model.zero_grad(set_to_none=True)
    loss.backward()
    grad_sign = x.grad.detach().sign()
    eps = _eps_px_to_norm(eps_px, device)
    x_adv = x - eps * grad_sign
    x_adv = _clamp_norm(x_adv).detach()
    return x_adv

def pgd_targeted(model, x_norm: torch.Tensor, y_target: torch.Tensor, eps_px: float = 8, alpha_px: float = 2, steps: int = 20, rand_init: bool = True, conf_stop: float = 0.90) -> torch.Tensor:
    device = x_norm.device
    eps = _eps_px_to_norm(eps_px, device)
    alpha = _eps_px_to_norm(alpha_px, device)
    x0 = x_norm.detach()
    if rand_init:
        x = x0 + torch.empty_like(x0).uniform_(-1, 1) * eps
        x = _clamp_norm(x)
    else:
        x = x0.clone()

    for _ in range(steps):
        x.requires_grad_(True)
        logits = model(x)
        loss = F.cross_entropy(logits, y_target)
        model.zero_grad(set_to_none=True); loss.backward()
        x = (x - alpha * x.grad.sign()).detach()   #
        x = torch.max(torch.min(x, x0 + eps), x0 - eps)
        x = _clamp_norm(x)

        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1)[0]
            if int(torch.argmax(probs)) == int(y_target) and float(probs[int(y_target)]) >= conf_stop:
                break
    return x

@st.cache_resource
def load_state(cfg_path: str):
    cfg = yaml.safe_load(open(cfg_path, "r"))
    ckpt = torch.load(cfg["model_out"], map_location="cpu")
    num_classes = len(ckpt["id2label"])

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    label2id = ckpt["label2id"]
    id2label = ckpt["id2label"]
    img_size = ckpt.get("img_size", 224)
    return cfg, model, img_size, label2id, id2label

def list_dataset_images(data_root: Path, limit=5000):
    img_dir = data_root / "Img" / "img_align_celeba"
    nested = img_dir / "img_align_celeba"
    if nested.exists():
        img_dir = nested
    return list(img_dir.glob("*.jpg"))[:limit]

@torch.no_grad()
def predict_full(model, x_norm: torch.Tensor):
    logits = model(x_norm)
    probs = torch.softmax(logits, dim=1)[0]
    pred = int(torch.argmax(probs))
    conf = float(probs[pred])
    return pred, conf, probs

def decide_access(pred_label: int, allowed_labels: set[int]):
    return ("allow", "Usuario autorizado") if pred_label in allowed_labels else ("deny", "Usuario no autorizado")

st.title("Software de control de acceso")

cfg, model, IMG_SIZE, label2id, id2label = load_state("config.yaml")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

st.sidebar.header("Config")

all_labels = sorted(list(label2id.keys()))
default_allowed = set(all_labels[:2])
allowed = st.sidebar.multiselect(
    "Usuarios autorizados (labels)", options=all_labels,
    default=list(default_allowed),
    help="Lista de clases (labels 0..C-1) con permiso."
)
allowed_labels = set(allowed) if allowed else set(default_allowed)

st.sidebar.subheader("Ataque adversarial dirigido")
atk_mode = st.sidebar.radio("Modo de ataque", ["FGSM", "PGD"], index=0)
default_target = sorted(list(allowed_labels))[0] if allowed_labels else all_labels[0]
target_label = st.sidebar.selectbox("Label objetivo (spoof)", options=all_labels,
                                    index=all_labels.index(default_target))
eps_px = st.sidebar.slider("ε (píxeles)", 0, 16, 8, 1)
if atk_mode == "PGD":
    alpha_px = st.sidebar.slider("α (píxeles)", 1, 8, 2, 1)
    steps = st.sidebar.slider("Pasos", 1, 50, 20, 1)
    conf_stop = st.sidebar.slider("Confianza objetivo (parada)", 0.5, 0.99, 0.90, 0.01)

tab1, tab2 = st.tabs(["Imagen del dataset", "Subir imagen"])

with tab1:
    st.subheader("Imágenes del dataset")
    imgs = list_dataset_images(Path(cfg["data_root"]), limit=200)
    f = st.selectbox("Selecciona una imagen", imgs)
    col_a, col_b = st.columns(2)

    if col_a.button("Verificar sin ataque"):
        img = Image.open(f).convert("RGB")
        x = to_tensor_and_norm(img, IMG_SIZE).to(device)
        pred, conf, probs = predict_full(model, x)
        decision, reason = decide_access(pred, allowed_labels)

        st.image(img, caption=f"Original: pred={pred}, conf={conf:.3f}", use_container_width=True)
        st.success(reason) if decision == "allow" else st.error(reason)

    if col_b.button("Aplicar ataque dirigido y verificar"):
        img = Image.open(f).convert("RGB")
        x = to_tensor_and_norm(img, IMG_SIZE).to(device)
        pred0, conf0, _ = predict_full(model, x)

        y_target = torch.tensor([int(target_label)], device=device, dtype=torch.long)
        if atk_mode == "FGSM":
            x_adv = fgsm_targeted(model, x, y_target, eps_px=eps_px)
        else:
            x_adv = pgd_targeted(model, x, y_target,
                                 eps_px=eps_px, alpha_px=alpha_px,
                                 steps=steps, rand_init=True, conf_stop=conf_stop)

        pred_adv, conf_adv, _ = predict_full(model, x_adv)

        dec0, reas0 = decide_access(pred0, allowed_labels)
        dec1, reas1 = decide_access(pred_adv, allowed_labels)

        c1, c2 = st.columns(2)
        c1.image(img, caption=f"Original → pred={pred0}, conf={conf0:.3f} — {reas0}", use_container_width=True)
        c2.image(to_pil_from01(denorm01(x_adv)), caption=f"{atk_mode} (ε={eps_px}px) → pred={pred_adv}, conf={conf_adv:.3f} — {reas1}", use_container_width=True)

with tab2:
    st.subheader("Sube una imagen y comprueba acceso")
    up = st.file_uploader("Elige imagen JPG/PNG", type=["jpg","jpeg","png"])
    if up:
        img = Image.open(up).convert("RGB")
        x = to_tensor_and_norm(img, IMG_SIZE).to(device)

        col1, col2 = st.columns(2)
        if col1.button("Verificar sin ataque", key="up_noatk"):
            pred, conf, probs = predict_full(model, x)
            decision, reason = decide_access(pred, allowed_labels)
            st.image(img, caption=f"Original: pred={pred}, conf={conf:.3f} — {reason}", use_container_width=True)

        if col2.button("Aplicar ataque dirigido y verificar", key="up_atk"):
            pred0, conf0, _ = predict_full(model, x)
            y_target = torch.tensor([int(target_label)], device=device, dtype=torch.long)
            if atk_mode == "FGSM":
                x_adv = fgsm_targeted(model, x, y_target, eps_px=eps_px)
            else:
                x_adv = pgd_targeted(model, x, y_target,
                                     eps_px=eps_px, alpha_px=alpha_px,
                                     steps=steps, rand_init=True, conf_stop=conf_stop)

            pred_adv, conf_adv, _ = predict_full(model, x_adv)

            dec0, reas0 = decide_access(pred0, allowed_labels)
            dec1, reas1 = decide_access(pred_adv, allowed_labels)

            c1, c2 = st.columns(2)
            c1.image(img, caption=f"Original → pred={pred0}, conf={conf0:.3f} — {reas0}", use_container_width=True)
            c2.image(to_pil_from01(denorm01(x_adv)),
                     caption=f"{atk_mode} (ε={eps_px}px) → pred={pred_adv}, conf={conf_adv:.3f} — {reas1}",
                     use_container_width=True)
