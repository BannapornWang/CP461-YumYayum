import os
import glob
import cv2
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

lr_dir = 'datasets/benchmark_gallery/LR'
out_base = 'results/Gallery'
os.makedirs(out_base, exist_ok=True)

# Select diverse test images for comparison
selected_files = [
    'anime_comic.jpg',   # Anime / Cartoon character (Degraded JPEG)
    'real_building.jpg', # Real-world Architecture / Texture
    'scenery_ost.png',   # Nature scenery (Trees, Foliage)
    'butterfly.png',     # Fine line patterns (Set5)
    'comic.png',         # Classic Manga / Comic drawing (Set14)
    'baboon.png'         # High-frequency hair texture (Set14)
]

def load_model(weights_path):
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    ckpt = torch.load(weights_path, map_location=device, weights_only=True)
    key = 'params_ema' if 'params_ema' in ckpt else ('params' if 'params' in ckpt else None)
    if key:
        model.load_state_dict(ckpt[key], strict=True)
    else:
        model.load_state_dict(ckpt, strict=True)
    model.eval()
    return model.to(device)

models = {
    'ESRGAN': 'weights/ESRGAN_x4.pth',
    'Real-ESRNet': 'weights/RealESRNet_x4plus.pth',
    'Real-ESRGAN (Official)': 'weights/RealESRGAN_x4plus.pth',
    'Real-ESRGAN (Team 52k)': 'net_g_52000.pth'
}

loaded_models = {}
for m_name, w_path in models.items():
    if os.path.exists(w_path):
        print(f"Loading {m_name}...")
        loaded_models[m_name] = load_model(w_path)

# Bicubic output folder
os.makedirs(os.path.join(out_base, 'Bicubic'), exist_ok=True)
for m_name in loaded_models.keys():
    os.makedirs(os.path.join(out_base, m_name), exist_ok=True)

for fname in selected_files:
    fpath = os.path.join(lr_dir, fname)
    if not os.path.exists(fpath):
        continue
    print(f"\nProcessing {fname}...")
    img = cv2.imread(fpath)
    h, w = img.shape[:2]
    
    # 1. Bicubic
    sr_bicubic = cv2.resize(img, (w * 4, h * 4), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(os.path.join(out_base, 'Bicubic', fname), sr_bicubic)

    # Convert to tensor for PyTorch models
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype('float32') / 255.0
    tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).to(device)

    # 2. Other Models
    for m_name, model in loaded_models.items():
        with torch.no_grad():
            sr_tensor = model(tensor)
            sr_np = sr_tensor.squeeze().float().cpu().clamp_(0, 1).numpy()
            sr_bgr = (sr_np.transpose(1, 2, 0)[:, :, [2, 1, 0]] * 255.0).round().astype('uint8')
            cv2.imwrite(os.path.join(out_base, m_name, fname), sr_bgr)

print("\nGallery inference completed!")

