import io
import numpy as np
from PIL import Image

from flask import Flask, request, render_template_string

import torch
import torch.nn as nn
from torchvision import models, transforms

app = Flask(__name__)

# ======================
# Config (sesuai training)
# ======================
MODEL_PATH = "batik_type_model.pth"

TARGET_W, TARGET_H = 540, 630
BLOCK_SIZE = 90
PATCH_SIZE = 180
EXPECTED_PATCHES = 30

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# mapping output (sesuai ImageFolder alphabetical): cap=0, tulis=1
CLASS_NAMES = ["Batik Cap", "Batik Tulis"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================
# Model
# ======================
def build_model(num_classes=2):
    # weights=None supaya tidak download pretrained di runtime
    model = models.vgg13_bn(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model

try:
    model = build_model(num_classes=2)
    state = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    print("✅ Model loaded:", MODEL_PATH, "| device:", DEVICE)
except Exception as e:
    print("❌ Failed to load model:", e)
    raise

# ======================
# Preprocess: resize -> 30 patches
# ======================
patch_tfms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

def resize_image(pil_img: Image.Image) -> Image.Image:
    pil_img = pil_img.convert("RGB")
    return pil_img.resize((TARGET_W, TARGET_H), resample=Image.BILINEAR)

def extract_30_patches(pil_img: Image.Image):
    w, h = pil_img.size
    blocks_x = w // BLOCK_SIZE  # 6
    blocks_y = h // BLOCK_SIZE  # 7

    patches = []
    for by in range(blocks_y - 1):
        for bx in range(blocks_x - 1):
            left = bx * BLOCK_SIZE
            upper = by * BLOCK_SIZE
            right = left + PATCH_SIZE
            lower = upper + PATCH_SIZE
            patches.append(pil_img.crop((left, upper, right, lower)))
    return patches

@torch.no_grad()
def predict_from_bytes(image_bytes: bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = resize_image(img)
    patches = extract_30_patches(img)

    if len(patches) != EXPECTED_PATCHES:
        raise RuntimeError(f"Patch count != {EXPECTED_PATCHES}. Got {len(patches)}")

    batch = torch.stack([patch_tfms(p) for p in patches], dim=0).to(DEVICE)  # [30,3,180,180]
    logits = model(batch)                       # [30,2]
    probs = torch.softmax(logits, dim=1)        # [30,2]
    avg_prob = probs.mean(dim=0).cpu().numpy()  # [2]

    pred_idx = int(np.argmax(avg_prob))
    return pred_idx, avg_prob

# ======================
# Routes (TAMPILAN DISAMAKAN DENGAN app.py (2))
# ======================
HOME_HTML = """
<div style="text-align:center; padding:50px;">
    <h1>Klasifikasi Jenis Batik</h1>
    <p>Upload gambar batik untuk mengklasifikasi jenis batik:</p>
    <p><b>Batik Cap</b> atau <b>Batik Tulis</b></p>
    <form action="/predict" method="post" enctype="multipart/form-data">
        <input type="file" name="file" accept="image/*" required><br><br>
        <button type="submit" style="padding:10px 20px; font-size:16px;">Prediksi</button>
    </form>
</div>
"""

RESULT_HTML = """
<div style="text-align:center; padding:50px;">
    <h1>Hasil Klasifikasi</h1>
    <h2 style="color:#2E7D32;">{{label}}</h2>

    <p style="margin-top:18px; margin-bottom:6px;"><b>Probabilitas rata-rata (30 patch)</b></p>
    <p style="margin:0;">Batik Cap: {{p_cap}}</p>
    <p style="margin:0;">Batik Tulis: {{p_tulis}}</p>

    <br>
    <a href="/" style="padding:10px 20px; background:#1976D2; color:white; text-decoration:none; border-radius:5px;">Kembali</a>
</div>
"""

@app.get("/")
def home():
    return render_template_string(HOME_HTML)

@app.post("/predict")
def predict_route():
    try:
        file = request.files["file"]
        pred_idx, avg_prob = predict_from_bytes(file.read())

        label = CLASS_NAMES[pred_idx]
        p_cap = float(avg_prob[0])
        p_tulis = float(avg_prob[1])

        return render_template_string(
            RESULT_HTML,
            label=label,
            p_cap=f"{p_cap:.4f}",
            p_tulis=f"{p_tulis:.4f}",
        )
    except Exception as e:
        return f"<h2 style='text-align:center; color:red;'>Error: {e}</h2><center><a href='/'>Kembali</a></center>"

if __name__ == "__main__":
    # Port 7860 wajib untuk HF Spaces
    app.run(host="0.0.0.0", port=7860)