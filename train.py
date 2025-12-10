from ultralytics import YOLO

if __name__ == "__main__":
    # Load a model
    model_yaml = "yolov8s_ddawa.yaml"
    data_yaml = "F:\DataFortraining\data\dataset_UAVDT.yaml"
    pre_model = "yolov8s.pt"

    model = YOLO(model_yaml, task='detect').load(pre_model)  # build from YAML and transfer weights

    # Train the model
    results = model.train(data=data_yaml, epochs=15, imgsz=640)