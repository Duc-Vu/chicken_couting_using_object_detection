from ultralytics import YOLO
import multiprocessing as mp

DATA = "dataset/yolo/data.yaml"
MODEL = "models/yolo/base/yolo26n.pt"

def main():
    model = YOLO(MODEL)

    model.train(
        data=DATA,
        epochs=100,
        imgsz=640,
        batch=16,
        device=0,
        optimizer="AdamW",
        lr0=1e-3,
        workers=8,
        patience=20,
    )

    metrics = model.val()
    print("mAP50:", metrics.box.map50)
    print("mAP50-95:", metrics.box.map)

    best = YOLO("runs/detect/train/weights/best.pt")
    best.export(format="onnx", opset=12)
    best.export(format="torchscript")

    best("test.jpg", conf=0.25, save=True)

if __name__ == "__main__":
    mp.freeze_support()
    main()