from ultralytics import YOLO

model = YOLO("D:/paper1/yolov12-main/best.pt")

# Run prediction
results = model.predict(
    source="D:/paper1/img/",  
    save_txt=True,                     # save YOLO-format .txt labels
    conf=0.6,
    imgsz=640,                  # inference size (pixels)
    augment=False,
    #agnostic_nms=True,
    nms=False,
    save=True,
    #save_crop=True                          # save output images with boxes (optional)
)

# Optional: print info about results
for r in results:
    print(f"Image: {r.path}")
    print(f"Detections: {r.boxes.shape}")  # shape: (num_boxes, 6) -> x1, y1, x2, y2, conf, cls
