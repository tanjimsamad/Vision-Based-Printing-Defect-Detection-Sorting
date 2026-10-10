import cv2
import numpy as np
import torch
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, ttk
from ultralytics import YOLO
import supervision as sv


# === Real YOLOv8 Detector with Supervision Integration ===
class ModelDetector:
    def __init__(self, model_path="best.pt", confidence_threshold=0.5):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"[INFO] Using device: {self.device}")
        self.model = YOLO(model_path).to(self.device)

        self.confidence_threshold = confidence_threshold
        self.box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()

    def detect_and_annotate(self, image):
        results = self.model(image)[0]
        detections = sv.Detections.from_ultralytics(results)

        # Filter based on confidence threshold
        detections = detections[detections.confidence > self.confidence_threshold]

        labels = [
            f"{class_name} {confidence:.2f}"
            for class_name, confidence in zip(detections['class_name'], detections.confidence)
        ]

        annotated_image = self.box_annotator.annotate(scene=image, detections=detections)
        annotated_image = self.label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

        return annotated_image


# === GUI Application ===
class DetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLO Object Detection GUI")
        self.running = False
        self.cap = None
        self.current_frame_image = None
        self.last_frame_shape = (480, 640, 3)  # Fallback default

        # 🔗 Initialize real detector
        self.detector = ModelDetector(model_path="best.pt")

        # === GUI Layout ===
        self.image_label = tk.Label(root)
        self.image_label.pack()

        button_frame = tk.Frame(root)
        button_frame.pack()

        self.start_button = tk.Button(button_frame, text="Start Webcam", command=self.start_webcam)
        self.start_button.grid(row=0, column=0)

        self.stop_button = tk.Button(button_frame, text="Stop Webcam", command=self.stop_webcam)
        self.stop_button.grid(row=0, column=1)

        self.conf_label = tk.Label(button_frame, text="Confidence:")
        self.conf_label.grid(row=0, column=2, padx=(10, 0))

        self.conf_slider = ttk.Scale(button_frame, from_=0.1, to=1.0, orient='horizontal', value=0.5,
                                     command=self.update_confidence)
        self.conf_slider.grid(row=0, column=3)

        self.report_button = tk.Button(root, text="Open Report", command=self.open_report)
        self.report_button.pack(pady=10)

        self.report_text = tk.Text(root, height=10)
        self.report_text.pack()

    def update_confidence(self, val):
        self.detector.confidence_threshold = float(val)

    def start_webcam(self):
        if not self.running:
            self.cap = cv2.VideoCapture(0)
            self.running = True
            self.process_video()

    def stop_webcam(self):
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None

        # Display black frame
        h, w, _ = self.last_frame_shape
        black = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.putText(black, "Webcam Stopped", (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 255, 255), 2)
        self.display_image(black)

    def process_video(self):
        if self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                self.last_frame_shape = frame.shape
                annotated = self.detector.detect_and_annotate(frame)
                self.display_image(annotated)
            self.root.after(15, self.process_video)

    def display_image(self, frame):
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(image)
        imgtk = ImageTk.PhotoImage(image=img)
        self.current_frame_image = imgtk
        self.image_label.config(image=self.current_frame_image)

    def open_report(self):
        file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if file_path:
            with open(file_path, 'r') as file:
                content = file.read()
            self.report_text.delete('1.0', tk.END)
            self.report_text.insert(tk.END, content)


# === Launch the Application ===
if __name__ == "__main__":
    root = tk.Tk()
    app = DetectionApp(root)
    root.mainloop()
# This code initializes a YOLOv8 detector with Supervision integration and provides a GUI for webcam-based object detection.
# The GUI allows users to start/stop the webcam, adjust the confidence threshold, and open a report file.
# The detector processes video frames, annotates detected objects, and displays the results in real-time.
# The application uses the `ultralytics` library for YOLOv8 and `supervision` for annotations.
# The GUI is built using `tkinter` and displays the annotated frames in a label.
# The application is designed to be user-friendly, with controls for starting/stopping the webcam and adjusting detection parameters.
# The code is structured to be modular, allowing for easy updates and modifications in the future.
# This code initializes a YOLOv8 detector with Supervision integration and provides a GUI for webcam-based object detection.
# The GUI allows users to start/stop the webcam, adjust the confidence threshold, and open a report file.
# The detector processes video frames, annotates detected objects, and displays the results in real-time.
# The application uses the `ultralytics` library for YOLOv8 and `supervision` for annotations.
# The GUI is built using `tkinter` and displays the annotated frames in a label.