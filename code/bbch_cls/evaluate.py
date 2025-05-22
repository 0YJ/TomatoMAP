import torch
import argparse
from models.classifiers import get_model
from datasets.custom_dataset import get_dataloaders
from utils import load_model

def evaluate_model(data_dir, num_classes, model_name='mobilenet_v3_large', 
                    model_path=None, batch_size=32, target_size=(640, 640)):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, test_loader = get_dataloaders(data_dir, batch_size, target_size, include_test=True)

    model = get_model(model_name, num_classes, pretrained=False).to(device)
    model_path = model_path or f"{model_name}_custom.pth"
    load_model(model, model_path, device)

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = correct / total
    print(f"Test accuracy of {model_name}: {acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--num_classes', type=int, default=50)
    parser.add_argument('--model_name', type=str, default='mobilenet_v3_large')
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--img_size', type=int, default=640)

    args = parser.parse_args()

    evaluate_model(
        data_dir=args.data_dir,
        num_classes=args.num_classes,
        model_name=args.model_name,
        model_path=args.model_path,
        batch_size=args.batch_size,
        target_size=(args.img_size, args.img_size)
    )
