import os
import urllib.request
import zipfile
import cv2

def download_file(url, target_path):
    if not os.path.exists(target_path):
        print(f"Downloading {url} to {target_path}...")
        urllib.request.urlretrieve(url, target_path)
        print("Done!")
    else:
        print(f"Already exists: {target_path}")

os.makedirs('weights', exist_ok=True)
os.makedirs('datasets/Set5/GT', exist_ok=True)
os.makedirs('datasets/Set5/LR_bicubic_x4', exist_ok=True)

# 1. Download pretrained weights
weights = {
    'RealESRGAN_x4plus.pth': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
    'RealESRNet_x4plus.pth': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth',
    # ESRGAN official pretrained weight
    'ESRGAN_x4.pth': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/ESRGAN_SRx4_DF2KOST_official-ff704c30.pth'
}

for name, url in weights.items():
    download_file(url, os.path.join('weights', name))

# 2. Download Set5 dataset
set5_urls = {
    'baby.png': 'https://raw.githubusercontent.com/jbhuang0604/SelfExSR/master/data/Set5/image_SRF_4/img_001_SRF_4_HR.png',
    'bird.png': 'https://raw.githubusercontent.com/jbhuang0604/SelfExSR/master/data/Set5/image_SRF_4/img_002_SRF_4_HR.png',
    'butterfly.png': 'https://raw.githubusercontent.com/jbhuang0604/SelfExSR/master/data/Set5/image_SRF_4/img_003_SRF_4_HR.png',
    'head.png': 'https://raw.githubusercontent.com/jbhuang0604/SelfExSR/master/data/Set5/image_SRF_4/img_004_SRF_4_HR.png',
    'woman.png': 'https://raw.githubusercontent.com/jbhuang0604/SelfExSR/master/data/Set5/image_SRF_4/img_005_SRF_4_HR.png'
}

print("\nDownloading Set5 HR images...")
for name, url in set5_urls.items():
    target = os.path.join('datasets/Set5/GT', name)
    try:
        download_file(url, target)
        # Create downscaled LR image (bicubic x4 downscale)
        img = cv2.imread(target)
        if img is not None:
            h, w = img.shape[:2]
            # Ensure dimensions divisible by 4
            h = (h // 4) * 4
            w = (w // 4) * 4
            img = img[:h, :w]
            cv2.imwrite(target, img)
            lr = cv2.resize(img, (w // 4, h // 4), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(os.path.join('datasets/Set5/LR_bicubic_x4', name), lr)
    except Exception as e:
        print(f"Error downloading {name}: {e}")

print("\nSetup datasets and weights complete!")

