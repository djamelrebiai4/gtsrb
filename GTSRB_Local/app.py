"""GTSRB Traffic Sign Classifier - Gradio Space
CNN (ResNet + Squeeze-Excitation) trained from scratch, 43 classes, 99.2 % test accuracy."""
import os
import numpy as np
import gradio as gr
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from PIL import Image, ImageOps
import spaces  # <-- Added for ZeroGPU compatibility

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "gtsrb_resnet.keras")
WEIGHTS_PATH = os.path.join(HERE, "best.weights.h5")
NUM_CLASSES = 43

# (English, العربية)
CLASSES = [
    ("Speed limit 20 km/h", "تحديد السرعة 20 كم/س"), ("Speed limit 30 km/h", "تحديد السرعة 30 كم/س"),
    ("Speed limit 50 km/h", "تحديد السرعة 50 كم/س"), ("Speed limit 60 km/h", "تحديد السرعة 60 كم/س"),
    ("Speed limit 70 km/h", "تحديد السرعة 70 كم/س"), ("Speed limit 80 km/h", "تحديد السرعة 80 كم/س"),
    ("End of speed limit 80 km/h", "نهاية تحديد السرعة 80 كم/س"), ("Speed limit 100 km/h", "تحديد السرعة 100 كم/س"),
    ("Speed limit 120 km/h", "تحديد السرعة 120 كم/س"), ("No passing", "ممنوع التجاوز"),
    ("No passing for vehicles over 3.5 t", "ممنوع التجاوز للمركبات فوق 3.5 طن"),
    ("Right-of-way at next intersection", "أولوية المرور عند التقاطع القادم"), ("Priority road", "طريق ذو أولوية"),
    ("Yield", "أعطِ الأولوية"), ("Stop", "قف"), ("No vehicles", "ممنوع مرور المركبات"),
    ("Vehicles over 3.5 t prohibited", "ممنوع الشاحنات فوق 3.5 طن"), ("No entry", "ممنوع الدخول"),
    ("General caution", "تحذير عام"), ("Dangerous curve left", "منعطف خطر إلى اليسار"),
    ("Dangerous curve right", "منعطف خطر إلى اليمين"), ("Double curve", "منعطف مزدوج"),
    ("Bumpy road", "طريق وعرة"), ("Slippery road", "طريق زلقة"),
    ("Road narrows on the right", "الطريق يضيق من اليمين"), ("Road work", "أشغال على الطريق"),
    ("Traffic signals", "إشارات المرور"), ("Pedestrians", "عبور المشاة"),
    ("Children crossing", "عبور الأطفال"), ("Bicycles crossing", "عبور الدراجات"),
    ("Beware of ice/snow", "احذر الجليد/الثلج"), ("Wild animals crossing", "عبور حيوانات برية"),
    ("End of all speed and passing limits", "نهاية جميع قيود السرعة والتجاوز"),
    ("Turn right ahead", "انعطف يميناً"), ("Turn left ahead", "انعطف يساراً"), ("Ahead only", "إلى الأمام فقط"),
    ("Go straight or right", "للأمام أو يميناً"), ("Go straight or left", "للأمام أو يساراً"),
    ("Keep right", "حافظ على اليمين"), ("Keep left", "حافظ على اليسار"),
    ("Roundabout mandatory", "دوار إجباري"), ("End of no passing", "نهاية منع التجاوز"),
    ("End of no passing (vehicles over 3.5 t)", "نهاية منع التجاوز للمركبات فوق 3.5 طن"),
]
LABELS = [f"{en} | {ar}" for en, ar in CLASSES]


# ------------------------------------------------------------------ Model
def _se(x, ratio=8):
    f = x.shape[-1]
    s = layers.GlobalAveragePooling2D()(x)
    s = layers.Dense(f // ratio, activation="relu")(s)
    s = layers.Dense(f, activation="sigmoid")(s)
    return layers.Multiply()([x, layers.Reshape((1, 1, f))(s)])


def _res(x, f, stride=1):
    sc = x
    x = layers.Conv2D(f, 3, strides=stride, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    x = layers.Conv2D(f, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = _se(x)
    if stride != 1 or sc.shape[-1] != f:
        sc = layers.Conv2D(f, 1, strides=stride, use_bias=False)(sc)
        sc = layers.BatchNormalization()(sc)
    return layers.ReLU()(layers.Add()([x, sc]))


def build_model(S=48):
    """Same architecture as in training (fallback if the .keras file can't be loaded)."""
    augment = keras.Sequential([
        layers.RandomRotation(0.06), layers.RandomZoom(0.12), layers.RandomTranslation(0.08, 0.08),
        layers.RandomContrast(0.2), layers.RandomBrightness(0.2, value_range=(0, 255)),
    ], name="augmentation")
    inp = keras.Input((S, S, 3))
    x = augment(inp)
    x = layers.Rescaling(1 / 255.0)(x)
    x = layers.Conv2D(32, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    x = _res(x, 32); x = _res(x, 64, 2); x = _res(x, 128, 2); x = _res(x, 256, 2)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    return keras.Model(inp, layers.Dense(NUM_CLASSES, activation="softmax")(x))


def load():
    try:
        return keras.models.load_model(MODEL_PATH, compile=False)
    except Exception as e:  # version mismatch -> rebuild + load weights
        print("load_model failed, falling back to weights:", repr(e))
        m = build_model(48)
        m.load_weights(WEIGHTS_PATH)
        return m


model = load()
S = int(model.input_shape[1])
print("Model ready. Input size:", S)


# ------------------------------------------------------------------ Inference
def preprocess(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB")
    w, h = img.size                                   # centre-crop to a square (photos are rarely square)
    m = min(w, h)
    img = img.crop(((w - m) // 2, (h - m) // 2, (w - m) // 2 + m, (h - m) // 2 + m))
    img = ImageOps.autocontrast(img, cutoff=1)         # same as in training
    return np.asarray(img.resize((S, S), Image.BILINEAR), dtype=np.uint8)


def _shift(X, dx, dy, pad=2):
    Xp = np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode="edge")
    return Xp[:, pad + dy: pad + dy + S, pad + dx: pad + dx + S]


# The @spaces.GPU decorator tells Hugging Face that this function can run on the GPU.
# Even though the model is small, this satisfies the ZeroGPU hardware requirement.
@spaces.GPU
def predict(img, use_tta):
    if img is None:
        return None, "", None
    x = preprocess(img)[None]
    views = [(0, 0), (2, 0), (-2, 0), (0, 2), (0, -2)] if use_tta else [(0, 0)]
    p = np.mean([model(_shift(x, dx, dy).astype("float32"), training=False).numpy() for dx, dy in views], axis=0)[0]
    top = int(p.argmax())
    conf = float(p[top])
    if conf >= 0.85:
        verdict = f"### ✅ {CLASSES[top][0]}\n### {CLASSES[top][1]}\nConfidence: **{conf:.1%}**"
    elif conf >= 0.5:
        verdict = f"### 🤔 Probably: {CLASSES[top][0]}\n{CLASSES[top][1]}\nConfidence only **{conf:.1%}** – try a sharper, closer photo."
    else:
        verdict = ("### ⚠️ Not sure\nConfidence is very low – this may not be one of the 43 German traffic signs, "
                   "or the sign is too small/blurry. Crop closely around the sign and try again.")
    seen = Image.fromarray(x[0]).resize((192, 192), Image.NEAREST)   # what the network actually sees
    return {LABELS[i]: float(p[i]) for i in range(NUM_CLASSES)}, verdict, seen


# ------------------------------------------------------------------ UI
EX_DIR = os.path.join(HERE, "examples")
examples = sorted(os.path.join(EX_DIR, f) for f in os.listdir(EX_DIR)) if os.path.isdir(EX_DIR) else []

INTRO = """
# 🚦 German Traffic Sign Recognition | التعرّف على إشارات المرور
Upload a photo of a traffic sign (or use your webcam) and a **CNN trained from scratch** will tell you what it is.
ارفع صورة لإشارة مرور (أو استخدم الكاميرا) وسيتعرّف عليها نموذج شبكة عصبونية التفافية مُدرَّب من الصفر.
"""

ABOUT = """
## About the model
| | |
|---|---|
| Dataset | GTSRB – German Traffic Sign Recognition Benchmark (43 classes, 39 209 train / 12 630 test images) |
| Architecture | Small ResNet with Squeeze-Excitation blocks, **1.26 M parameters** |
| Training | Adam, Cross-Entropy (label smoothing 0.05), warm-up + cosine LR, **10 epochs**, data augmentation, **no pre-trained weights** |
| Test accuracy | **99.22 %** |
| Macro F1-score | **98.85 %** |
| Speed | ≈ 0.2 ms per image (batch inference, CPU/GPU) |

## Good to know
- The model knows **only the 43 German signs** of GTSRB. Other countries' signs (or non-signs) can get a confident-looking but wrong answer.
- It works best on a **close crop** of a single sign. Photos are centre-cropped to a square automatically.
- *Test-time augmentation* (5 slightly shifted views averaged) is on by default and was chosen on a validation set.

Built for the **AI Model Challenge 1 – ACISN'26, University of Bouira**.
"""

with gr.Blocks(title="GTSRB Traffic Sign Classifier") as demo:
    gr.Markdown(INTRO)
    with gr.Tab("🔍 Classify"):
        with gr.Row():
            with gr.Column():
                inp = gr.Image(type="pil", label="Traffic sign photo", sources=["upload", "webcam", "clipboard"])
                tta = gr.Checkbox(value=True, label="Test-time augmentation (slower, more robust)")
                btn = gr.Button("Classify", variant="primary")
            with gr.Column():
                verdict = gr.Markdown()
                lab = gr.Label(num_top_classes=5, label="Top-5 predictions")
                seen = gr.Image(label="What the network sees (48×48)", interactive=False, height=200)
        if examples:
            gr.Examples(examples=[[e] for e in examples], inputs=[inp], label="Try an example")
        for trig in (btn.click, inp.change, tta.change):
            trig(predict, [inp, tta], [lab, verdict, seen])
    with gr.Tab("ℹ️ About"):
        gr.Markdown(ABOUT)

if __name__ == "__main__":
    # ssr_mode=False prevents the Node.js proxy crash on Hugging Face Spaces
    demo.launch(ssr_mode=False)