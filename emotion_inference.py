# emotion_inference.py
# Loads fine-tuned ViT once and provides a helper to classify one image.

import torch
from io import BytesIO
from PIL import Image
import numpy as np
from transformers import AutoImageProcessor, AutoModelForImageClassification

BASE_PROC_ID = "google/vit-base-patch16-224-in21k"
# MODEL_DIR = "best_model_vit"  # folder with config.json + model.safetensors # dont use this now
MODEL_ID = "murk1sh/best_model_vit"


def _ensure_pil_rgb(img):
    """
    Accepts: PIL.Image | np.ndarray (HWC or CHW) | bytes/bytearray | str path
    Returns: PIL.Image in RGB mode
    """
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    if isinstance(img, (bytes, bytearray)):
        return Image.open(BytesIO(img)).convert("RGB")
    if isinstance(img, np.ndarray):
        # If CHW, convert to HWC
        if img.ndim == 3 and img.shape[0] in (1, 3):
            img = np.moveaxis(img, 0, 2)
        # If grayscale, expand to 3 channels
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        return Image.fromarray(img.astype(np.uint8)).convert("RGB")
    if isinstance(img, str):
        return Image.open(img).convert("RGB")
    raise TypeError("Unsupported image type for inference")

def load_emotion_model(model_id: str = MODEL_ID):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Processor handles resize to 224, normalization, and tensor conversion
    try:
        processor = AutoImageProcessor.from_pretrained(BASE_PROC_ID)
    except Exception:
        # Fallback to base model name if processor wasn't saved with the fine-tuned dir
        processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

    model = AutoModelForImageClassification.from_pretrained(model_id).to(device)
    model.eval()

    # Pull label names from config if present; otherwise fallback to FER-7 common set
    id2label = getattr(model.config, "id2label", None)
    if id2label and len(id2label) == model.config.num_labels:
        class_names = [id2label[str(i)] if str(i) in id2label else id2label[i]
                       for i in range(model.config.num_labels)]
    else:
        class_names = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

    return model, processor, class_names, device

@torch.no_grad()
def infer_emotions(img, model, processor, class_names, device, top_k: int = 3):
    """
    Runs a forward pass on one image and returns:
      {
        "label": top_label,
        "scores": {label: prob, ...},   # all classes
        "topk": [(label, prob), ...]    # top_k predictions (if top_k>0)
      }
    'img' can be path/PIL/np.ndarray/bytes. np.ndarray can be gray or color.
    """
    pil_img = _ensure_pil_rgb(img)
    inputs = processor(images=pil_img, return_tensors="pt").to(device)
    logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

    scores = {label: float(p) for label, p in zip(class_names, probs)}
    top_idx = int(probs.argmax())
    top_label = class_names[top_idx]

    topk = None
    if top_k and top_k > 0:
        idxs = probs.argsort()[::-1][:top_k]
        topk = [(class_names[i], float(probs[i])) for i in idxs]

    return {"label": top_label, "scores": scores, "topk": topk}
