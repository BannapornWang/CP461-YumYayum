import os
import urllib.request
import cv2

os.makedirs('datasets/benchmark_gallery/LR', exist_ok=True)

# 1. Download Set14 sample images
set14_urls = {
    'comic.png': 'https://raw.githubusercontent.com/jbhuang0604/SelfExSR/master/data/Set14/image_SRF_4/img_004_SRF_4_LR.png',
    'baboon.png': 'https://raw.githubusercontent.com/jbhuang0604/SelfExSR/master/data/Set14/image_SRF_4/img_001_SRF_4_LR.png',
    'monarch.png': 'https://raw.githubusercontent.com/jbhuang0604/SelfExSR/master/data/Set14/image_SRF_4/img_011_SRF_4_LR.png',
    'lenna.png': 'https://raw.githubusercontent.com/jbhuang0604/SelfExSR/master/data/Set14/image_SRF_4/img_010_SRF_4_LR.png',
    'pepper.png': 'https://raw.githubusercontent.com/jbhuang0604/SelfExSR/master/data/Set14/image_SRF_4/img_013_SRF_4_LR.png'
}

for name, url in set14_urls.items():
    dst = os.path.join('datasets/benchmark_gallery/LR', name)
    if not os.path.exists(dst):
        try:
            print(f"Downloading {name}...")
            urllib.request.urlretrieve(url, dst)
        except Exception as e:
            print(f"Error {name}: {e}")

# 2. Add real-world input images from Real-ESRGAN official repo
repo_inputs = {
    'anime_comic.jpg': 'Real-ESRGAN-repo/inputs/0030.jpg',
    'real_building.jpg': 'Real-ESRGAN-repo/inputs/0014.jpg',
    'scenery_ost.png': 'Real-ESRGAN-repo/inputs/OST_009.png',
    'interior.jpg': 'Real-ESRGAN-repo/inputs/ADE_val_00000114.jpg'
}

for name, src in repo_inputs.items():
    if os.path.exists(src):
        img = cv2.imread(src)
        if img is not None:
            # save as jpg/png
            cv2.imwrite(os.path.join('datasets/benchmark_gallery/LR', name), img)
            print(f"Added {name} from repo inputs")

# 3. Add Set5 images (baby, butterfly, bird)
for f in os.listdir('datasets/Set5/LR_bicubic_x4'):
    if f.endswith('.png'):
        img = cv2.imread(os.path.join('datasets/Set5/LR_bicubic_x4', f))
        cv2.imwrite(os.path.join('datasets/benchmark_gallery/LR', f), img)

print("Gallery LR dataset ready! Total images:", len(os.listdir('datasets/benchmark_gallery/LR')))

