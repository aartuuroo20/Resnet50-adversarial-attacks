import io
import yaml
import numpy as np
from pathlib import Path
from PIL import Image, ImageFilter

import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import models, transforms

st.set_page_config(page_title="Software de control de acceso", layout="wide")

# ---------------------------- Utilidades de normalización ----------------------------

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)

def to_tensor_and_norm(pil_img, size):
    tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    return tf(pil_img).unsqueeze(0)

def entropy(probs: torch.Tensor) -> float:
    p = probs + 1e-12
    return float(-(p * torch.log(p)).sum())

def denorm01(x_norm: torch.Tensor) -> torch.Tensor:
    """De normalizado a [0,1] (para mostrar/guardar)."""
    return (x_norm * IMAGENET_STD.to(x_norm.device) + IMAGENET_MEAN.to(x_norm.device)).clamp(0,1)

def to_pil_from01(x01: torch.Tensor) -> Image.Image:
    x = (x01.detach().cpu().squeeze(0).permute(1,2,0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(x)

def _eps_px_to_norm(eps_px: float, device: torch.device) -> torch.Tensor:
    """Convierte ε en píxeles (0..255) a ε por canal en espacio normalizado (1,3,1,1)."""
    return (float(eps_px) / 255.0) / IMAGENET_STD.to(device)

def _clamp_norm(x_norm: torch.Tensor) -> torch.Tensor:
    """Clampea en espacio [0,1] y vuelve a normalizado para mantener rango válido."""
    x01 = denorm01(x_norm)
    return (x01 - IMAGENET_MEAN.to(x_norm.device)) / IMAGENET_STD.to(x_norm.device)

# --------------------------------- FGSM dirigido ------------------------------------

def fgsm_targeted(model, x_norm: torch.Tensor, y_target: torch.Tensor, eps_px: float) -> torch.Tensor:
    """
    FGSM dirigido: empuja la imagen hacia la clase objetivo.
    Fórmula: x_adv = x - eps * sign(∇_x CE(model(x), y_target))
             (resta el signo del gradiente para minimizar la pérdida de la clase objetivo)
    - x_norm: imagen normalizada (B,3,H,W)
    - y_target: labels objetivo (B,)
    - eps_px: epsilon en píxeles (escala [0..255])
    """
    device = x_norm.device
    x = x_norm.clone().detach().requires_grad_(True)
    logits = model(x)
    loss = F.cross_entropy(logits, y_target)  # minimizar CE hacia la clase objetivo
    model.zero_grad(set_to_none=True)
    loss.backward()
    grad_sign = x.grad.detach().sign()

    eps = _eps_px_to_norm(eps_px, device)  # (1,3,1,1)
    x_adv = x - eps * grad_sign            # targeted: "menos" gradiente
    x_adv = _clamp_norm(x_adv).detach()
    return x_adv

# --------------------------------- Carga de modelo ----------------------------------

@st.cache_resource
def load_state(cfg_path: str):
    cfg = yaml.safe_load(open(cfg_path, "r"))
    ckpt = torch.load(cfg["model_out"], map_location="cpu")
    num_classes = len(ckpt["id2label"])

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    label2id = ckpt["label2id"]                 # label -> identity_id_original
    id2label = ckpt["id2label"]                 # identity_id_original -> label
    img_size = ckpt.get("img_size", 224)
    return cfg, model, img_size, label2id, id2label

def list_dataset_images(data_root: Path, limit=5000):
    img_dir = data_root / "Img" / "img_align_celeba"
    nested = img_dir / "img_align_celeba"
    if nested.exists():
        img_dir = nested
    paths = list(img_dir.glob("*.jpg"))
    return paths[:limit]

# ---------------------------------- Defensas ----------------------------------------

def jpeg_compress(pil_img: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    return Image.open(buf).convert("RGB")

def gaussian_blur(pil_img: Image.Image, sigma: float) -> Image.Image:
    if sigma <= 0:
        return pil_img
    return pil_img.filter(ImageFilter.GaussianBlur(radius=float(sigma)))

def bit_depth_reduction(pil_img: Image.Image, bits: int) -> Image.Image:
    bits = int(bits)
    if bits >= 8:
        return pil_img
    arr = np.array(pil_img).astype(np.float32)
    levels = 2 ** bits
    arr = np.floor(arr / 255.0 * (levels - 1) + 0.5) / (levels - 1) * 255.0
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def apply_defenses(pil_img: Image.Image, *, enable_jpeg=False, jpeg_q=90,
                   enable_blur=False, blur_sigma=0.0, enable_bits=False, bits=8) -> Image.Image:
    out = pil_img
    if enable_jpeg:
        out = jpeg_compress(out, jpeg_q)
    if enable_blur:
        out = gaussian_blur(out, blur_sigma)
    if enable_bits:
        out = bit_depth_reduction(out, bits)
    return out

# ---------------------------------- Predicción y decisión ---------------------------------

@torch.no_grad()
def predict_full(model, x_norm: torch.Tensor):
    logits = model(x_norm)
    probs = torch.softmax(logits, dim=1)[0]
    pred = int(torch.argmax(probs))
    conf = float(probs[pred])
    return pred, conf, probs

def decide_access(pred_label: int, allowed_labels: set[int]):
    if pred_label in allowed_labels:
        return "allow", "Usuario autorizado"
    return "deny", "Usuario no autorizado"

# -------------------------------------- UI -----------------------------------------

st.title("Software de control de acceso")

cfg, model, IMG_SIZE, label2id, id2label = load_state("config.yaml")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# --- barra lateral: política, defensas y ataque ---
st.sidebar.header("Config")

# (1) Lista de usuarios autorizados (labels del modelo, no identity_id)
all_labels = sorted(list(label2id.keys()))
default_allowed = set(all_labels[:2])  # por defecto, los dos primeros labels del subset
allowed = st.sidebar.multiselect(
    "Usuarios autorizados (labels)", options=all_labels, default=list(default_allowed),
    help="Lista de clases (labels 0..C-1) con permiso."
)
allowed_labels = set(allowed) if allowed else set(default_allowed)

# (2) Defensas (preprocesado)
st.sidebar.subheader("Defensas")
use_jpeg = st.sidebar.checkbox("JPEG compression", value=False)
jpeg_q = st.sidebar.slider("Calidad JPEG", 10, 100, 85) if use_jpeg else 85

use_blur = st.sidebar.checkbox("Gaussian blur", value=False)
blur_sigma = st.sidebar.slider("Sigma blur", 0.0, 3.0, 0.6, 0.1) if use_blur else 0.0

use_bits = st.sidebar.checkbox("Bit-depth reduction", value=False)
bits = st.sidebar.slider("Bits", 1, 8, 8) if use_bits else 8

# (3) Ataque dirigido (FGSM)
st.sidebar.subheader("Ataque adversarial (dirigido, FGSM)")
enable_attack = st.sidebar.checkbox("Activar ataque FGSM dirigido", value=False)
# por defecto, objetivo = primer autorizado (o el 1º label si no hay selección)
default_target = sorted(list(allowed_labels))[0] if allowed_labels else all_labels[0]
target_label = st.sidebar.selectbox("Label objetivo (spoof)", options=all_labels, index=all_labels.index(default_target))
eps_px = st.sidebar.slider("ε (píxeles)", 0, 16, 8, 1)

# ------------------- MAIN APP -------------------

tab1, tab2 = st.tabs(["Imagen del dataset", "Subir imagen propia"])

# --- Tab1: elegir una imagen del dataset ---
with tab1:
    st.subheader("Prueba con imágenes del dataset CelebA")
    imgs = list_dataset_images(Path(cfg["data_root"]), limit=200)
    f = st.selectbox("Selecciona una imagen", imgs)
    col_a, col_b = st.columns(2)
    if col_a.button("Verificar SIN ataque"):
        img = Image.open(f).convert("RGB")
        img_def = apply_defenses(img, enable_jpeg=use_jpeg, jpeg_q=jpeg_q,
                                      enable_blur=use_blur, blur_sigma=blur_sigma,
                                      enable_bits=use_bits, bits=bits)
        x = to_tensor_and_norm(img_def, IMG_SIZE).to(device)
        pred, conf, probs = predict_full(model, x)
        decision, reason = decide_access(pred, allowed_labels)

        st.image(img, caption=f"Original: pred={pred} conf={conf:.3f}", use_container_width=True)
        if decision == "allow":
            st.success(reason)
        else:
            st.error(reason)

    if col_b.button("Aplicar ATAQUE DIRIGIDO y verificar"):
        img = Image.open(f).convert("RGB")
        img_def = apply_defenses(img, enable_jpeg=use_jpeg, jpeg_q=jpeg_q,
                                      enable_blur=use_blur, blur_sigma=blur_sigma,
                                      enable_bits=use_bits, bits=bits)
        x = to_tensor_and_norm(img_def, IMG_SIZE).to(device)

        # predicción base (info)
        pred0, conf0, _ = predict_full(model, x)

        # FGSM dirigido al label seleccionado
        y_target = torch.tensor([int(target_label)], device=device, dtype=torch.long)
        x_adv = fgsm_targeted(model, x, y_target, eps_px=eps_px)
        pred_adv, conf_adv, _ = predict_full(model, x_adv)

        # Decisiones
        dec0, reas0 = decide_access(pred0, allowed_labels)
        dec1, reas1 = decide_access(pred_adv, allowed_labels)

        # Mostrar lado a lado
        c1, c2 = st.columns(2)
        c1.image(img, caption=f"Original → pred={pred0}, conf={conf0:.3f} — {('✅ '+reas0) if dec0=='allow' else ('⛔ '+reas0)}",
                 use_container_width=True)
        c2.image(to_pil_from01(denorm01(x_adv)), caption=f"Adversarial (FGSM ε={eps_px}px) → pred={pred_adv}, conf={conf_adv:.3f} — {('✅ '+reas1) if dec1=='allow' else ('⛔ '+reas1)}",
                 use_container_width=True)

# --- Tab2: subir imagen externa ---
with tab2:
    st.subheader("Sube una imagen y comprueba acceso")
    up = st.file_uploader("Elige imagen JPG/PNG", type=["jpg","jpeg","png"])
    if up:
        img = Image.open(up).convert("RGB")
        img_def = apply_defenses(img, enable_jpeg=use_jpeg, jpeg_q=jpeg_q,
                                      enable_blur=use_blur, blur_sigma=blur_sigma,
                                      enable_bits=use_bits, bits=bits)
        x = to_tensor_and_norm(img_def, IMG_SIZE).to(device)

        col1, col2 = st.columns(2)
        if col1.button("Verificar SIN ataque", key="up_noatk"):
            pred, conf, probs = predict_full(model, x)
            decision, reason = decide_access(pred, allowed_labels)
            st.image(img, caption=f"Original: pred={pred}, conf={conf:.3f} — {('✅ '+reason) if decision=='allow' else ('⛔ '+reason)}",
                     use_container_width=True)

        if col2.button("Aplicar ATAQUE DIRIGIDO y verificar", key="up_atk"):
            pred0, conf0, _ = predict_full(model, x)
            y_target = torch.tensor([int(target_label)], device=device, dtype=torch.long)
            x_adv = fgsm_targeted(model, x, y_target, eps_px=eps_px)
            pred_adv, conf_adv, _ = predict_full(model, x_adv)

            dec0, reas0 = decide_access(pred0, allowed_labels)
            dec1, reas1 = decide_access(pred_adv, allowed_labels)

            c1, c2 = st.columns(2)
            c1.image(img, caption=f"Original → pred={pred0}, conf={conf0:.3f} — {('✅ '+reas0) if dec0=='allow' else ('⛔ '+reas0)}",
                     use_container_width=True)
            c2.image(to_pil_from01(denorm01(x_adv)), caption=f"Adversarial (FGSM ε={eps_px}px) → pred={pred_adv}, conf={conf_adv:.3f} — {('✅ '+reas1) if dec1=='allow' else ('⛔ '+reas1)}",
                     use_container_width=True)
