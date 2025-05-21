import torch
import argparse
from models.classifiers import get_model
from datasets.custom_dataset import get_dataloaders
from utils import load_model
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import os
from PIL import ImageDraw, ImageFont
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pandas as pd

def denormalize(img_tensor, mean, std):
    mean = torch.tensor(mean).view(3, 1, 1).to(img_tensor.device)
    std = torch.tensor(std).view(3, 1, 1).to(img_tensor.device)
    return torch.clamp(img_tensor * std + mean, 0, 1)

def get_font(size=30, bold=False):
    font_paths = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]
    for path in font_paths:
        try:
            return ImageFont.truetype(path, size=size)
        except:
            continue
    print("no font. using default")
    return ImageFont.load_default()


def imshow(img_tensor, save_path=None,
           gt_label=None, pred_label=None, correct=True, prob=None,
           mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    img = img_tensor.cpu().clone()
    img = denormalize(img, mean, std)
    img = transforms.ToPILImage()(img)

    draw = ImageDraw.Draw(img)
    font = get_font(size=40, bold=True)

    draw.text((100, 10), f"GT: {gt_label}", fill="white", font=font)

    prob_text = f" ({prob:.1f}%)" if prob is not None else ""
    pred_text = f"Pred: {pred_label}{prob_text}" + ("" if correct else "X")
    draw.text((100, 50), pred_text, fill="red" if not correct else "yellow", font=font)

    if save_path:
        img.save(save_path)


def save_confusion_matrix(y_true, y_pred, class_names, save_path_img, save_path_csv):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))

    fig, ax = plt.subplots(figsize=(14, 12))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, xticks_rotation=45, cmap='jet', colorbar=True, include_values=False)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path_img)
    plt.close()

    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
    df_cm.to_csv(save_path_csv)


def evaluate_model(data_dir, num_classes, model_name='mobilenet_v3_large',
                   model_path=None, batch_size=32, target_size=(640, 640),
                   show_images=True, save_images=False, save_dir='eval_results'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, test_loader = get_dataloaders(data_dir, batch_size, target_size, include_test=True)

    model = get_model(model_name, num_classes, pretrained=False).to(device)
    model_path = model_path or f"{model_name}_custom.pth"
    load_model(model, model_path, device)

    model.eval()
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    class_names = test_loader.dataset.classes

    if save_images and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with torch.no_grad():
        for idx, (images, labels) in enumerate(tqdm(test_loader, desc="Evaluating")):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            conf_scores, predicted = torch.max(probs, dim=1)

            all_preds.extend(predicted.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            if show_images or save_images:
                for i in range(images.size(0)):
                    pred_label = predicted[i].item()
                    true_label = labels[i].item()
                    pred_class = class_names[pred_label]
                    true_class = class_names[true_label]
                    conf = conf_scores[i].item() * 100

                    is_correct = pred_label == true_label
                    filename = f"{idx}_{i}_pred_{pred_class}_gt_{true_class}.png"
                    save_path = os.path.join(save_dir, filename) if save_images else None

                    if save_images:
                        imshow(images[i],
                               save_path=save_path,
                               gt_label=true_class,
                               pred_label=pred_class,
                               correct=is_correct,
                               prob=conf)

    acc = correct / total
    print(f"\n Test Accuracy for {model_name}: {acc:.4f}")

    if save_images:
        cm_img_path = os.path.join(save_dir, "confusion_matrix.png")
        cm_csv_path = os.path.join(save_dir, "confusion_matrix.csv")
        save_confusion_matrix(all_labels, all_preds, class_names, cm_img_path, cm_csv_path)
        print(f"Confusion matrix saved to: {cm_img_path}")
        print(f"Confusion matrix CSV saved to: {cm_csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--num_classes', type=int, default=50)
    parser.add_argument('--model_name', type=str, default='mobilenet_v3_large')
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--img_size', type=int, default=640)
    parser.add_argument('--show_images', action='store_true', help='show image?')
    parser.add_argument('--save_images', action='store_true', help='save image?')
    parser.add_argument('--save_dir', type=str, default='eval_results', help='save dir for output')

    args = parser.parse_args()

    evaluate_model(
        data_dir=args.data_dir,
        num_classes=args.num_classes,
        model_name=args.model_name,
        model_path=args.model_path,
        batch_size=args.batch_size,
        target_size=(args.img_size, args.img_size),
        show_images=args.show_images,
        save_images=args.save_images,
        save_dir=args.save_dir
    )
