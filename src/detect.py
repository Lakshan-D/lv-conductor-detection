import argparse
from ultralytics import YOLO

def run(image_path):
    model = YOLO('runs/detect/runs/train/lv_conductor_v1-2/weights/best.pt')
    results = model(image_path)
    for r in results:
        r.save(filename='results/output.jpg')
    print("Detection complete. Saved to results/output.jpg")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True)
    args = parser.parse_args()
    run(args.image)