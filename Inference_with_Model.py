import cv2
import supervision as sv
from ultralytics import YOLO
import torch

class ModelDetector:
    def __init__(self, model_path="best.pt"):
        # Set device to 'cuda' if available, otherwise 'cpu'
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        # Load the model and move it to the selected device
        self.model = YOLO(model_path).to(self.device)
        print(f"Model is on device: {next(self.model.parameters()).device}")
        
        # Initialize annotators once
        self.box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()
    
    def detect_and_annotate(self, image):
        # Run inference on the selected device (model handles input automatically)
        results = self.model(image)[0]
        detections = sv.Detections.from_ultralytics(results)

        # Generate labels for annotations
        labels = [
            f"{class_name} {confidence:.2f}"
            for class_name, confidence
            in zip(detections['class_name'], detections.confidence)
        ]

        # Annotate the image using instance annotators
        annotated_image = self.box_annotator.annotate(scene=image, detections=detections)
        annotated_image = self.label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
        
        return annotated_image
    
    def process_image(self, image_path):
        # Process a single image
        image = cv2.imread(image_path)
        annotated_image = self.detect_and_annotate(image)
        cv2.imshow("YOLO Detection", annotated_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def process_webcam(self, camera_index=0): # Default to camera index 1
        # Process webcam feed
        cap = cv2.VideoCapture(camera_index)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            annotated_frame = self.detect_and_annotate(frame)
            cv2.imshow("YOLO Detection", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = ModelDetector()
    
    # Uncomment the required mode
    # detector.process_image("images/boiled1.jpg")
    detector.process_webcam()