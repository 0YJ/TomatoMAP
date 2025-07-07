from ultralytics import YOLO

model = YOLO("./best.pt")

# Run prediction
results = model.predict(
    source="D:/paper1/img/",  
    save_txt=True,
    conf=0.6,
    imgsz=640,
    augment=False,
    #agnostic_nms=True,
    nms=False,
    save=True,
    #save_crop=True
)

#for r in results:
#    print(f"Image: {r.path}")
#    print(f"Detections: {r.boxes.shape}")  # shape: (num_boxes, 6) -> x1, y1, x2, y2, conf, cls
