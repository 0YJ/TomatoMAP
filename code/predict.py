import torch
import argparse
from models.classifiers import get_model
from datasets.custom_dataset import get_dataloaders
from utils import load_model
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import os


def imshow(img_tensor, title=None, save_path=None):
    """显示或保存一张图像"""
    img = img_tensor.cpu().clone()
    img = transforms.ToPILImage()(img)
    plt.imshow(img)
    if title:
        plt.title(title)
    plt.axis('off')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


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

    # ✅ 获取真实类别名列表
    class_names = test_loader.dataset.classes

    if save_images and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with torch.no_grad():
        for idx, (images, labels) in enumerate(tqdm(test_loader, desc="Evaluating")):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # ✅ 显示每张图像及预测/真实标签
            if show_images or save_images:
                for i in range(images.size(0)):
                    pred_label = predicted[i].item()
                    true_label = labels[i].item()
                    pred_class = class_names[pred_label]
                    true_class = class_names[true_label]

                    is_correct = pred_label == true_label
                    mark = "" if is_correct else " ❌"
                    title = f"BBCH: {pred_class} (GT: {true_class}){mark}"

                    filename = f"{idx}_{i}_pred_{pred_class}_gt_{true_class}.png"
                    save_path = os.path.join(save_dir, filename) if save_images else None
                    imshow(images[i], title=title, save_path=save_path)

    acc = correct / total
    print(f"\n✅ Test Accuracy for {model_name}: {acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--num_classes', type=int, default=50)
    parser.add_argument('--model_name', type=str, default='mobilenet_v3_large')
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--img_size', type=int, default=640)
    parser.add_argument('--show_images', action='store_true', help='是否显示图片')
    parser.add_argument('--save_images', action='store_true', help='是否保存图片')
    parser.add_argument('--save_dir', type=str, default='eval_results', help='保存预测图像的文件夹')

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
