import cv2
import os
import numpy as np
from PIL import Image, ImageEnhance
import random

# === Root dataset folder ===
dataset_root = "dataset"

def apply_augmentations(img, idx, original_name, output_folder):
    augmented = []

    # 1. Blur
    augmented.append(cv2.GaussianBlur(img, (5, 5), 0))

    # 2. Bright & Dark
    pil_img = Image.fromarray(img)
    bright = np.array(ImageEnhance.Brightness(pil_img).enhance(1.5))
    dark = np.array(ImageEnhance.Brightness(pil_img).enhance(0.5))
    augmented.extend([bright, dark])

    # 3. Rotation
    for angle in [-10, 10]:
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
        rotated = cv2.warpAffine(img, M, (w, h))
        augmented.append(rotated)

    # 4. Flip
    augmented.append(cv2.flip(img, 1))

    # 5. Random Crop (Zoom)
    h, w = img.shape[:2]
    crop_size = int(0.9 * min(h, w))
    x = random.randint(0, w - crop_size)
    y = random.randint(0, h - crop_size)
    crop = img[y:y+crop_size, x:x+crop_size]
    augmented.append(cv2.resize(crop, (w, h)))

    # 6. Noise
    noise = img + np.random.normal(0, 25, img.shape).astype(np.uint8)
    augmented.append(np.clip(noise, 0, 255))

    # Save all
    for i, aug in enumerate(augmented):
        out_name = f"{os.path.splitext(original_name)[0]}_aug{i+1}_{idx}.jpg"
        out_path = os.path.join(output_folder, out_name)
        cv2.imwrite(out_path, aug)

# === Process all folders ===
for person_folder in os.listdir(dataset_root):
    input_path = os.path.join(dataset_root, person_folder)
    if os.path.isdir(input_path) and not person_folder.endswith("_augmented"):
        output_path = os.path.join(dataset_root, f"{person_folder}_augmented")
        os.makedirs(output_path, exist_ok=True)

        print(f"📁 Processing {person_folder}...")

        for idx, filename in enumerate(os.listdir(input_path)):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(input_path, filename)
                img = cv2.imread(img_path)

                if img is not None:
                    apply_augmentations(img, idx, filename, output_path)

print("✅ Done! All people’s augmented images are ready.")
