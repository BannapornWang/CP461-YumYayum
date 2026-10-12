import os
import glob
import cv2
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

lr_dir = 'datasets/eval_20/LR'
img_paths = sorted(glob.glob(os.path.join(lr_dir, '*.png')))

# Create albums matching the requested structure
album_dirs = {
    'Bicubic': 'results/Albums/Bicubic',
    'Ours': 'results/Albums/Ours',
    'RealESRGAN': 'results/Albums/RealESRGAN'
}

for d in album_dirs.values():
    os.makedirs(d, exist_ok=True)

# 1. Bicubic
print("\n--- Generating Bicubic Album ---")
for p in img_paths:
    name = os.path.basename(p)
    img = cv2.imread(p)
    h, w = img.shape[:2]
    sr = cv2.resize(img, (w * 4, h * 4), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(os.path.join(album_dirs['Bicubic'], name), sr)
print(f"Bicubic album complete! ({len(img_paths)} images)")

def load_rrdb_model(weights_path):
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    ckpt = torch.load(weights_path, map_location=device, weights_only=True)
    key = 'params_ema' if 'params_ema' in ckpt else ('params' if 'params' in ckpt else None)
    if key:
        model.load_state_dict(ckpt[key], strict=True)
    else:
        model.load_state_dict(ckpt, strict=True)
    model.eval()
    return model.to(device)

def run_inference_for_album(model, target_dir, album_name):
    print(f"\n--- Generating {album_name} Album ---")
    for p in img_paths:
        name = os.path.basename(p)
        img = cv2.imread(p, cv2.IMREAD_COLOR).astype('float32') / 255.0
        img = img[:, :, [2, 1, 0]]
        tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(tensor)
            output = output.squeeze().float().cpu().clamp_(0, 1).numpy()
        output = (output.transpose(1, 2, 0)[:, :, [2, 1, 0]] * 255.0).round().astype('uint8')
        cv2.imwrite(os.path.join(target_dir, name), output)
    print(f"{album_name} album complete! ({len(img_paths)} images)")

# 2. Ours (net_g_52000.pth)
if os.path.exists('net_g_52000.pth'):
    ours_model = load_rrdb_model('net_g_52000.pth')
    run_inference_for_album(ours_model, album_dirs['Ours'], 'Ours (Team 52k)')

# 3. RealESRGAN Official
if os.path.exists('weights/RealESRGAN_x4plus.pth'):
    realesrgan_model = load_rrdb_model('weights/RealESRGAN_x4plus.pth')
    run_inference_for_album(realesrgan_model, album_dirs['RealESRGAN'], 'RealESRGAN (Official)')

print("\nAll 3 albums generated successfully!")

