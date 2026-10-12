import os
import urllib.request
import cv2

os.makedirs('datasets/eval_20/GT', exist_ok=True)
os.makedirs('datasets/eval_20/LR', exist_ok=True)

# 20 diverse test images (Set5 + Set14 + BSDS100 / Urban samples)
image_sources = {
    # Set5 (5 images)
    '01_baby.png': 'https://raw.githubusercontent.com/jbhuang0604/SelfExSR/master/data/Set5/image_SRF_4/img_001_SRF_4_HR.png',
    '02_bird.png': 'https://raw.githubusercontent.com/jbhuang0604/SelfExSR/master/data/Set5/image_SRF_4/img_002_SRF_4_HR.png',
    '03_butterfly.png': 'https://raw.githubusercontent.com/jbhuang0604/SelfExSR/master/data/Set5/image_SRF_4/img_003_SRF_4_HR.png',
    '04_head.png': 'https://raw.githubusercontent.com/jbhuang0604/SelfExSR/master/data/Set5/image_SRF_4/img_004_SRF_4_HR.png',
    '05_woman.png': 'https://raw.githubusercontent.com/jbhuang0604/SelfExSR/master/data/Set5/image_SRF_4/img_005_SRF_4_HR.png',
    # Set14 (14 images)
    '06_baboon.png': 'https://raw.githubusercontent.com/jbhuang0604/SelfExSR/master/data/Set14/image_SRF_4/img_001_SRF_4_HR.png',
    '07_barbara.png': 'https://raw.githubusercontent.com/jbhuang0604/SelfExSR/master/data/Set14/image_SRF_4/img_002_SRF_4_HR.png',
    '08_bridge.png': 'https://raw.githubusercontent.com/jbhuang0604/SelfExSR/master/data/Set14/image_SRF_4/img_003_SRF_4_HR.png',
    '09_coastguard.png': 'https://raw.githubusercontent.com/jbhuang0604/SelfExSR/master/data/Set14/image_SRF_4/img_004_SRF_4_HR.png',
    '10_comic.png': 'https://raw.githubusercontent.com/jbhuang0604/SelfExSR/master/data/Set14/image_SRF_4/img_005_SRF_4_HR.png',
    '11_face.png': 'https://raw.githubusercontent.com/jbhuang0604/SelfExSR/master/data/Set14/image_SRF_4/img_006_SRF_4_HR.png',
    '12_flowers.png': 'https://raw.githubusercontent.com/jbhuang0604/SelfExSR/master/data/Set14/image_SRF_4/img_007_SRF_4_HR.png',
    '13_foreman.png': 'https://raw.githubusercontent.com/jbhuang0604/SelfExSR/master/data/Set14/image_SRF_4/img_008_SRF_4_HR.png',
    '14_lenna.png': 'https://raw.githubusercontent.com/jbhuang0604/SelfExSR/master/data/Set14/image_SRF_4/img_009_SRF_4_HR.png',
    '15_man.png': 'https://raw.githubusercontent.com/jbhuang0604/SelfExSR/master/data/Set14/image_SRF_4/img_010_SRF_4_HR.png',
    '16_monarch.png': 'https://raw.githubusercontent.com/jbhuang0604/SelfExSR/master/data/Set14/image_SRF_4/img_011_SRF_4_HR.png',
    '17_monkey.png': 'https://raw.githubusercontent.com/jbhuang0604/SelfExSR/master/data/Set14/image_SRF_4/img_012_SRF_4_HR.png',
    '18_pepper.png': 'https://raw.githubusercontent.com/jbhuang0604/SelfExSR/master/data/Set14/image_SRF_4/img_013_SRF_4_HR.png',
    '19_ppt3.png': 'https://raw.githubusercontent.com/jbhuang0604/SelfExSR/master/data/Set14/image_SRF_4/img_014_SRF_4_HR.png',
}

print("Downloading and preparing 20 benchmark images...")
for name, url in image_sources.items():
    gt_p = os.path.join('datasets/eval_20/GT', name)
    lr_p = os.path.join('datasets/eval_20/LR', name)
    if not os.path.exists(gt_p):
        try:
            urllib.request.urlretrieve(url, gt_p)
            img = cv2.imread(gt_p)
            if img is not None:
                h, w = img.shape[:2]
                h = (h // 4) * 4
                w = (w // 4) * 4
                img = img[:h, :w]
                cv2.imwrite(gt_p, img)
                lr = cv2.resize(img, (w // 4, h // 4), interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(lr_p, lr)
                print(f"Downloaded & downscaled: {name}")
        except Exception as e:
            print(f"Error {name}: {e}")

# 20th image: Use high-res OST foliage from repo inputs
ost_src = 'Real-ESRGAN-repo/inputs/OST_009.png'
if os.path.exists(ost_src):
    img = cv2.imread(ost_src)
    h, w = img.shape[:2]
    h, w = (h // 4) * 4, (w // 4) * 4
    img = img[:h, :w]
    cv2.imwrite('datasets/eval_20/GT/20_foliage.png', img)
    lr = cv2.resize(img, (w // 4, h // 4), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite('datasets/eval_20/LR/20_foliage.png', lr)
    print("Added 20_foliage.png")

print("Dataset preparation complete! Total files:", len(os.listdir('datasets/eval_20/LR')))

