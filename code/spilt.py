import os
import shutil
import random
from sklearn.model_selection import train_test_split

random.seed(42)

train_dir = 'data/train'
val_dir = 'data/val'

os.makedirs(val_dir, exist_ok=True)

class_names = sorted(os.listdir(train_dir))

for class_name in class_names:
    class_path = os.path.join(train_dir, class_name)
    if not os.path.isdir(class_path):
        continue

    images = os.listdir(class_path)
    images = [img for img in images if img.lower().endswith(('.jpg', '.png', '.jpeg'))]

    # 分层划分：80% train, 20% val
    train_imgs, val_imgs = train_test_split(images, test_size=0.2, random_state=42)

    val_class_dir = os.path.join(val_dir, class_name)
    os.makedirs(val_class_dir, exist_ok=True)

    for img in val_imgs:
        src = os.path.join(class_path, img)
        dst = os.path.join(val_class_dir, img)
        shutil.move(src, dst)

    print(f"{class_name}: total={len(images)}, moved to val={len(val_imgs)}")

print("\n✅ 分层划分完成！")