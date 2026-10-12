import os
import glob
import cv2
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

lr_dir = 'datasets/eval_20/LR'
img_paths = sorted(glob.glob(os.path.join(lr_dir, '*.png')))

album_dirs = {
    'Bicubic': 'results/Albums/Bicubic',
    'ESRGAN': 'results/Albums/ESRGAN',
    'BSRGAN': 'results/Albums/BSRGAN',
    'Real-ESRNet': 'results/Albums/Real-ESRNet',
    'Ours': 'results/Albums/Ours',
    'RealESRGAN': 'results/Albums/RealESRGAN'
}

for d in album_dirs.values():
    os.makedirs(d, exist_ok=True)

# 1. Bicubic
for p in img_paths:
    name = os.path.basename(p)
    out_p = os.path.join(album_dirs['Bicubic'], name)
    if not os.path.exists(out_p):
        img = cv2.imread(p)
        h, w = img.shape[:2]
        sr = cv2.resize(img, (w * 4, h * 4), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(out_p, sr)

def convert_bsrgan_keys(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        new_k = k
        new_k = new_k.replace('RRDB_trunk.', 'body.')
        new_k = new_k.replace('.RDB1.', '.rdb1.')
        new_k = new_k.replace('.RDB2.', '.rdb2.')
        new_k = new_k.replace('.RDB3.', '.rdb3.')
        new_k = new_k.replace('trunk_conv.', 'conv_body.')
        new_k = new_k.replace('HRconv.', 'conv_hr.')
        new_k = new_k.replace('conv_last.', 'conv_last.')
        new_state_dict[new_k] = v
    return new_state_dict

def load_rrdb_model(weights_path, is_bsrgan=False):
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    if is_bsrgan:
        ckpt = convert_bsrgan_keys(ckpt)
        model.load_state_dict(ckpt, strict=False)
    else:
        key = 'params_ema' if 'params_ema' in ckpt else ('params' if 'params' in ckpt else None)
        if key:
            model.load_state_dict(ckpt[key], strict=True)
        else:
            model.load_state_dict(ckpt, strict=True)
    model.eval()
    return model.to(device)

def run_model(model_name, weight_path, is_bsrgan=False):
    print(f"\n--- Processing {model_name} ---")
    target_dir = album_dirs[model_name]
    model = load_rrdb_model(weight_path, is_bsrgan=is_bsrgan)
    for p in img_paths:
        name = os.path.basename(p)
        out_p = os.path.join(target_dir, name)
        if os.path.exists(out_p):
            continue
        img = cv2.imread(p, cv2.IMREAD_COLOR).astype('float32') / 255.0
        img = img[:, :, [2, 1, 0]]
        tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(tensor)
            output = output.squeeze().float().cpu().clamp_(0, 1).numpy()
        output = (output.transpose(1, 2, 0)[:, :, [2, 1, 0]] * 255.0).round().astype('uint8')
        cv2.imwrite(out_p, output)
    print(f"{model_name} done! ({len(os.listdir(target_dir))} images)")

# Run models
if os.path.exists('weights/ESRGAN_x4.pth'):
    run_model('ESRGAN', 'weights/ESRGAN_x4.pth')

if os.path.exists('weights/BSRGAN.pth'):
    run_model('BSRGAN', 'weights/BSRGAN.pth', is_bsrgan=True)

if os.path.exists('weights/RealESRNet_x4plus.pth'):
    run_model('Real-ESRNet', 'weights/RealESRNet_x4plus.pth')

if os.path.exists('net_g_52000.pth'):
    run_model('Ours', 'net_g_52000.pth')

if os.path.exists('weights/RealESRGAN_x4plus.pth'):
    run_model('RealESRGAN', 'weights/RealESRGAN_x4plus.pth')

print("\nAll 6 models finished generating all 20 images!")

