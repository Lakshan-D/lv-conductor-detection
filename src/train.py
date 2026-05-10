from ultralytics import YOLO

model = YOLO('yolov8n.pt')

results = model.train(
    data='Power-Line-3/data.yaml',
    epochs=15,
    imgsz=640,
    batch=8,
    name='lv_conductor_v1',
    project='runs/train'
)

print("Training complete.")