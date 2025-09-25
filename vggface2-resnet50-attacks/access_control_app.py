import yaml
import streamlit as st
import torch
import torch.nn.functional as F

from pathlib import Path
from PIL import Image
from torchvision import models, transforms

def load_id2name_csv(meta_csv_path: str):
    """
    Devuelve {Class_ID -> Name} leyendo identity_meta.csv de forma robusta:
    - quita espacios en cabeceras y valores
    - ignora líneas corruptas
    - case-insensitive para nombres de columna (soporta 'Class_ID', 'class_ID', 'class_id', etc.)
    """
    from pathlib import Path
    p = Path(meta_csv_path)
    if not p.exists():
        print(f"[meta] No existe CSV en {p}")
        return {}

    # ---- Intento con pandas (engine=python, tolerante) ----
    try:
        import pandas as pd
        df = pd.read_csv(
            p,
            engine="python",
            sep=", ",
            quotechar='"',
            skipinitialspace=True,  # quita espacios tras coma
            encoding="utf-8-sig",
            on_bad_lines="skip",
        )
        # Normaliza cabeceras: strip + lower (guardamos mapping al original por si acaso)
        orig_cols = list(df.columns)
        norm_cols = [str(c).strip() for c in orig_cols]
        lower_map = {c.lower(): c for c in norm_cols}  # lower -> normalizada
        df.columns = norm_cols

        # Buscar columnas por lower()
        id_key_lower   = next((k for k in ["class_id","id"] if k in lower_map), None)
        name_key_lower = next((k for k in ["name","identity","person"] if k in lower_map), None)
        if not id_key_lower or not name_key_lower:
            print(f"[meta] Columnas no encontradas. Cabeceras: {norm_cols}")
            raise KeyError("missing columns")

        id_col = lower_map[id_key_lower]
        nm_col = lower_map[name_key_lower]

        # Limpia valores
        df[id_col] = df[id_col].astype(str).str.strip()
        df[nm_col] = df[nm_col].astype(str).str.strip()

        id2name = dict(zip(df[id_col], df[nm_col]))
        print(f"[meta] (pandas) Cargados {len(id2name)} nombres")
        return id2name
    except Exception as e:
        print(f"[meta] pandas falló: {e}")

    # ---- Fallback: csv.DictReader ----
    try:
        import csv
        id2name = {}
        with p.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f, skipinitialspace=True)
            if not reader.fieldnames:
                print("[meta] csv sin cabeceras.")
                return {}
            # normaliza cabeceras
            norm_names = [s.strip() for s in reader.fieldnames]
            lower_map = {s.lower(): s for s in norm_names}
            id_key_lower   = next((k for k in ["class_id","id"] if k in lower_map), None)
            name_key_lower = next((k for k in ["name","identity","person"] if k in lower_map), None)
            if not id_key_lower or not name_key_lower:
                print(f"[meta] Columnas no encontradas en {norm_names}")
                return {}
            id_key = lower_map[id_key_lower]
            nm_key = lower_map[name_key_lower]

            for row in reader:
                # re-strip por si acaso
                cid = str(row.get(id_key, "")).strip()
                nm  = str(row.get(nm_key, "")).strip()
                if cid:
                    id2name[cid] = nm or cid

        print(f"[meta] (csv.DictReader) Cargados {len(id2name)} nombres")
        return id2name
    except Exception as e2:
        print(f"[meta] csv.DictReader falló: {e2}")
        return {}



def build_label_maps(label2id: dict, meta_csv_path: str | None = None):
    """
    Devuelve:
      - label2display: label(int) -> nombre bonito (real si CSV, si no 'n000xxx')
      - display2label: nombre bonito -> label(int), para UI
    """
    id2name = load_id2name_csv(meta_csv_path) if meta_csv_path else {}
    label2display = {}
    for lbl, ident in label2id.items():
        pretty = id2name.get(str(ident), ident)  # 'Aaron_Eckhart' o 'n000123'
        label2display[int(lbl)] = pretty
    display2label = {v: k for k, v in label2display.items()}
    return label2display, display2label

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

    # NUEVO: mapas de nombre
    META_CSV = "data/vggface2hq/meta/identity_meta.csv"  # si no existe, fallback a 'n000xxx'
    label2display, display2label = build_label_maps(label2id, META_CSV)

    return cfg, model, img_size, label2id, id2label, label2display, display2label


def list_dataset_images_vgg(cfg, split="train", limit=5000):
    base = Path(cfg["train_dir"] if split=="train" else cfg["test_dir"])
    paths = list(base.glob("*/*.jpg"))  # carpeta identidad/*.jpg
    return paths[:limit]


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

cfg, model, IMG_SIZE, label2id, id2label, LABEL2DISPLAY, DISPLAY2LABEL = load_state("config.yaml")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

st.sidebar.header("Config")

all_labels = sorted(list(label2id.keys()))
all_names = [LABEL2DISPLAY[lbl] for lbl in all_labels]
default_allowed_labels = set(all_labels[:2])
default_allowed_names = [LABEL2DISPLAY[lbl] for lbl in sorted(list(default_allowed_labels))]

allowed_names = st.sidebar.multiselect(
    "Usuarios autorizados",
    options=all_names,
    default=default_allowed_names,
    help="Clases permitidas (se muestran por nombre)."
)
# Convertimos nombres -> labels
allowed_labels = {DISPLAY2LABEL[name] for name in allowed_names} if allowed_names else default_allowed_labels

st.sidebar.subheader("Ataque adversarial dirigido")
atk_mode = st.sidebar.radio("Modo de ataque", ["FGSM", "PGD"], index=0)

default_target_label = sorted(list(allowed_labels))[0] if allowed_labels else all_labels[0]
default_target_name = LABEL2DISPLAY[default_target_label]
target_name = st.sidebar.selectbox(
    "Usuario objetivo (spoof)",
    options=all_names,
    index=all_names.index(default_target_name)
)
target_label = DISPLAY2LABEL[target_name]

eps_px = st.sidebar.slider("ε (píxeles)", 0, 16, 8, 1)
if atk_mode == "PGD":
    alpha_px = st.sidebar.slider("α (píxeles)", 1, 8, 2, 1)
    steps = st.sidebar.slider("Pasos", 1, 50, 20, 1)
    conf_stop = st.sidebar.slider("Confianza objetivo (parada)", 0.5, 0.99, 0.90, 0.01)

tab1, tab2 = st.tabs(["Imagen del dataset", "Subir imagen"])

with tab1:
    st.subheader("Imágenes del dataset")
    imgs = list_dataset_images_vgg(cfg, split="train", limit=200)
    f = st.selectbox("Selecciona una imagen", imgs)
    col_a, col_b = st.columns(2)

    if col_a.button("Verificar sin ataque"):
        img = Image.open(f).convert("RGB")
        x = to_tensor_and_norm(img, IMG_SIZE).to(device)
        pred, conf, probs = predict_full(model, x)
        pred_name = LABEL2DISPLAY.get(pred, str(pred))   # <<< nombre bonito
        decision, reason = decide_access(pred, allowed_labels)

        st.image(img, caption=f"Original: pred={pred_name}, conf={conf:.3f}", use_container_width=True)
        st.success(reason) if decision == "allow" else st.error(reason)

    if col_b.button("Aplicar ataque dirigido y verificar"):
        img = Image.open(f).convert("RGB")
        x = to_tensor_and_norm(img, IMG_SIZE).to(device)
        pred0, conf0, _ = predict_full(model, x)
        pred0_name = LABEL2DISPLAY.get(pred0, str(pred0))   # <<< nombre bonito

        y_target = torch.tensor([int(target_label)], device=device, dtype=torch.long)
        if atk_mode == "FGSM":
            x_adv = fgsm_targeted(model, x, y_target, eps_px=eps_px)
        else:
            x_adv = pgd_targeted(model, x, y_target,
                                 eps_px=eps_px, alpha_px=alpha_px,
                                 steps=steps, rand_init=True, conf_stop=conf_stop)

        pred_adv, conf_adv, _ = predict_full(model, x_adv)
        pred_adv_name = LABEL2DISPLAY.get(pred_adv, str(pred_adv))   # <<< nombre bonito

        dec0, reas0 = decide_access(pred0, allowed_labels)
        dec1, reas1 = decide_access(pred_adv, allowed_labels)

        c1, c2 = st.columns(2)
        c1.image(img, caption=f"Original → pred={pred0_name}, conf={conf0:.3f} — {reas0}", use_container_width=True)
        c2.image(to_pil_from01(denorm01(x_adv)),
                 caption=f"{atk_mode} (ε={eps_px}px) → pred={pred_adv_name}, conf={conf_adv:.3f} — {reas1}",
                 use_container_width=True)

with tab2:
    st.subheader("Sube una imagen y comprueba acceso")
    up = st.file_uploader("Elige imagen JPG/PNG", type=["jpg","jpeg","png"])
    if up:
        img = Image.open(up).convert("RGB")
        x = to_tensor_and_norm(img, IMG_SIZE).to(device)

        col1, col2 = st.columns(2)
        if col1.button("Verificar sin ataque", key="up_noatk"):
            pred, conf, probs = predict_full(model, x)
            pred_name = LABEL2DISPLAY.get(pred, str(pred))   # <<< nombre bonito
            decision, reason = decide_access(pred, allowed_labels)
            st.image(img, caption=f"Original: pred={pred_name}, conf={conf:.3f} — {reason}", use_container_width=True)

        if col2.button("Aplicar ataque dirigido y verificar", key="up_atk"):
            pred0, conf0, _ = predict_full(model, x)
            pred0_name = LABEL2DISPLAY.get(pred0, str(pred0))   # <<< nombre bonito

            y_target = torch.tensor([int(target_label)], device=device, dtype=torch.long)
            if atk_mode == "FGSM":
                x_adv = fgsm_targeted(model, x, y_target, eps_px=eps_px)
            else:
                x_adv = pgd_targeted(model, x, y_target,
                                     eps_px=eps_px, alpha_px=alpha_px,
                                     steps=steps, rand_init=True, conf_stop=conf_stop)

            pred_adv, conf_adv, _ = predict_full(model, x_adv)
            pred_adv_name = LABEL2DISPLAY.get(pred_adv, str(pred_adv))   # <<< nombre bonito

            dec0, reas0 = decide_access(pred0, allowed_labels)
            dec1, reas1 = decide_access(pred_adv, allowed_labels)

            c1, c2 = st.columns(2)
            c1.image(img, caption=f"Original → pred={pred0_name}, conf={conf0:.3f} — {reas0}", use_container_width=True)
            c2.image(to_pil_from01(denorm01(x_adv)),
                     caption=f"{atk_mode} (ε={eps_px}px) → pred={pred_adv_name}, conf={conf_adv:.3f} — {reas1}",
                     use_container_width=True)

