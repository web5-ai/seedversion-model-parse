from ultralytics import YOLO

class yolo_model:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)
    def detect(self, image):
        results = self.model(image)
        return results
