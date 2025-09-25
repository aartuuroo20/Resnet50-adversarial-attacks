import streamlit as st
import torch
import yaml
import random
import os
import pandas as pd

from torchvision import transforms, models
from PIL import Image
from pathlib import Path

st.set_page_config(page_title="CelebA ResNet-50", page_icon="🖼️")

@st.cache_resource
def load_state(cfg_path):
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

def predict(model, tf, img: Image.Image):
    x = tf(img).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        pred = int(torch.argmax(probs))
        conf = float(probs[pred])
    return pred, conf

def sample_image(celeba_root: Path):
    img_dir = celeba_root / "Img" / "img_align_celeba"
    files = list(img_dir.glob("*.jpg"))
    return random.choice(files) if files else None

st.title("Reconocimiento de personas con CelebA y ResNet‑50")

cfg, ckpt, model, tf = load_state("config.yaml")
label2id = ckpt["label2id"]  
id2label = ckpt["id2label"]  
label2name = {lab: str(identity) for lab, identity in label2id.items()}

tab1, tab2 = st.tabs(["Imagen aleatoria del dataset", "Subir una imagen"])

with tab1:
    st.write("Muestra aleatoria del dataset y predicción del modelo entrenado.")
    if st.button("Cargar imagen aleatoria"):
        f = sample_image(Path(cfg["data_root"]))
        if f is None:
            st.error("No encuentro imágenes. ¿Has ejecutado el entrenamiento para descargar CelebA?")
        else:
            img = Image.open(f).convert("RGB")
            st.image(img, caption=f.name, use_container_width=True)
            pred, conf = predict(model, tf, img)
            st.success(f"Predicción: identidad (label) = {pred} | identity_id = {label2name[pred]} | confianza = {conf:.3f}")

with tab2:
    up = st.file_uploader("Sube una imagen (JPG/PNG)", type=["jpg","jpeg","png"])
    if up:
        img = Image.open(up).convert("RGB")
        st.image(img, caption="Imagen subida", use_container_width=True)
        pred, conf = predict(model, tf, img)
        st.info(f"Predicción: identidad (label) = {pred} | identity_id = {label2name[pred]} | confianza = {conf:.3f}")

st.caption("Nota: Este modelo está entrenado solo sobre un subconjunto de identidades (top-K por frecuencia) para fines de demo.")
