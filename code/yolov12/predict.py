from ultralytics import YOLO

model = YOLO("D:/EOC/yolov12-main/best.pt")

# Run prediction
results = model.predict(
    source="D:/EOC/img/",
    save_txt=True,
    conf=0.6,
    imgsz=640,
    augment=False,
    #agnostic_nms=True,
    nms=False,
    save=True,
    #save_crop=True
)

for r in results:
    print(f"Image: {r.path}")
    print(f"Detections: {r.boxes.shape}")
