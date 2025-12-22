import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

def load_dinov2_model(device="cuda"):
    """Load DINOv2 model with proper configuration."""
    model_name = "facebook/dinov2-base"
    
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    
    return processor, model

def dinov2_embedding(np_img: np.ndarray, model, processor, device="cuda"):
    """Compute DINOv2 embeddings. Expects np_img in [0,1] range."""
    if np_img.max() <= 1.0:
        img = Image.fromarray((np_img * 255).astype(np.uint8))
    else:
        img = Image.fromarray(np_img.astype(np.uint8))
    
    inputs = processor(images=img, return_tensors="pt").to(device)
    print(f"[DEBUG] Input tensor shape to DINOv2: {inputs['pixel_values'].shape}")
    
    with torch.no_grad():
        outputs = model(**inputs)
        feats = outputs.last_hidden_state[:, 1:, :]  
    
    num_patches = feats.shape[1]
    dim = feats.shape[2]
    h = w = int(num_patches ** 0.5)
    
    emb = feats[0].T.reshape(1, dim, h, w)
    return emb

def compute_dinov2_similarity(img1: np.ndarray, img2: np.ndarray, 
                              model, processor, device="cuda"):
    """Compute cosine similarity map between two images."""
    emb1 = dinov2_embedding(img1, model, processor, device)
    emb2 = dinov2_embedding(img2, model, processor, device)
    
    norm1 = torch.nn.functional.normalize(emb1, p=2, dim=1)
    norm2 = torch.nn.functional.normalize(emb2, p=2, dim=1)
    sim = torch.nn.functional.cosine_similarity(norm1, norm2, dim=1).squeeze(0)
    
    return sim.cpu().numpy()
