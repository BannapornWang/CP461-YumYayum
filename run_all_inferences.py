import os
import glob
import cv2
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Input images (Set5 LR)
lr_dir = 'datasets/Set5/LR_bicubic_x4'
gt_dir = 'datasets/Set5/GT'
output_base = 'results/Set5'
os.makedirs(output_base, exist_ok=True)

img_paths = sorted(glob.glob(os.path.join(lr_dir, '*.png')))

def run_bicubic():
    print("\n--- Generating Bicubic outputs ---")
    out_dir = os.path.join(output_base, 'Bicubic')
    os.makedirs(out_dir, exist_ok=True)
    for p in img_paths:
        name = os.path.basename(p)
        img = cv2.imread(p)
        h, w = img.shape[:2]
        sr = cv2.resize(img, (w * 4, h * 4), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(os.path.join(out_dir, name), sr)
    print("Bicubic done!")

def load_rrdb_model(weights_path):
    # Standard RRDBNet architecture (used by ESRGAN, Real-ESRNet, Real-ESRGAN x4plus)
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    checkpoint = torch.load(weights_path, map_location=device, weights_only=True)
    if 'params_ema' in checkpoint:
        keyname = 'params_ema'
    elif 'params' in checkpoint:
        keyname = 'params'
    else:
        keyname = None
    
    if keyname:
        model.load_state_dict(checkpoint[keyname], strict=True)
    else:
        model.load_state_dict(checkpoint, strict=True)
    model.eval()
    model = model.to(device)
    return model

def run_model_inference(model_name, weight_path):
    print(f"\n--- Generating {model_name} outputs ---")
    out_dir = os.path.join(output_base, model_name)
    os.makedirs(out_dir, exist_ok=True)
    
    model = load_rrdb_model(weight_path)
    
    for p in img_paths:
        name = os.path.basename(p)
        img = cv2.imread(p, cv2.IMREAD_COLOR).astype('float32') / 255.0
        # BGR to RGB, HWC to CHW
        img = img[:, :, [2, 1, 0]]
        tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(tensor)
            output = output.squeeze().float().cpu().clamp_(0, 1).numpy()
        
        # CHW to HWC, RGB to BGR
        output = (output.transpose(1, 2, 0)[:, :, [2, 1, 0]] * 255.0).round().astype('uint8')
        cv2.imwrite(os.path.join(out_dir, name), output)
    print(f"{model_name} done!")

# 1. Bicubic
run_bicubic()

# 2. ESRGAN
if os.path.exists('weights/ESRGAN_x4.pth'):
    run_model_inference('ESRGAN', 'weights/ESRGAN_x4.pth')

# 3. Real-ESRNet
if os.path.exists('weights/RealESRNet_x4plus.pth'):
    run_model_inference('Real-ESRNet', 'weights/RealESRNet_x4plus.pth')

# 4. Real-ESRGAN Official
if os.path.exists('weights/RealESRGAN_x4plus.pth'):
    run_model_inference('Real-ESRGAN (Official)', 'weights/RealESRGAN_x4plus.pth')

# 5. Real-ESRGAN (Team model - 52000 iter)
if os.path.exists('net_g_52000.pth'):
    run_model_inference('Real-ESRGAN_Team', 'net_g_52000.pth')

print("\nAll inferences complete!")

