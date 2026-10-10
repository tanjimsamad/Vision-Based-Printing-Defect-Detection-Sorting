# Please connect to the Arduino board before running this script.

import cv2
import numpy as np
import torch
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk
from ultralytics import YOLO
from pyfirmata2 import Arduino, SERVO
import threading
from time import sleep, time, strftime

class VideoProcessor:
    def __init__(self, confidence_threshold=0.5, log_func=None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = YOLO("best.pt").to(self.device)
        print(f"Model is running on device: {self.model.device}")

        try:
            self.board = Arduino("COM3")  # Change as needed
        except Exception as e:
            raise Exception(f"Failed to connect to Arduino: {e}")

        self.pin = 9
        self.board.digital[self.pin].mode = SERVO
        self.INITIALANGLE = 0
        self.board.digital[self.pin].write(self.INITIALANGLE)

        self.confidence_threshold = confidence_threshold
        self.last_trigger_time = 0
        self.cooldown = 1.0
        self.log_func = log_func  # Function to send log messages

    def process_frame(self, frame):
        result = self.model(frame)
        class_name = None
        center_point = None

        height, width, _ = frame.shape
        line_y = int(height * 0.6)
        cv2.line(frame, (0, line_y), (width, line_y), (0, 255, 0), 2)

        for box in result[0].boxes:
            x1, y1, x2, y2 = [round(x) for x in box.xyxy[0].tolist()]
            center_point = ((x1 + x2) // 2, (y1 + y2) // 2)
            probability = box.conf[0].item()
            if probability < self.confidence_threshold:
                continue
            class_id = int(box.cls[0].item())
            class_name = result[0].names[class_id]

            box_color = (0, 0, 255) if class_name == "Miss_Print" else (255, 0, 0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(frame, f"{class_name} - {probability:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

            if class_name == "Miss_Print" and y1 <= line_y <= y2:
                current_time = time()
                if current_time - self.last_trigger_time > self.cooldown:
                    self.last_trigger_time = current_time
                    if self.log_func:
                        self.log_func(f"Miss_Print detected at position {center_point}")
                    threading.Thread(target=self.trigger_servo).start()

        return frame, class_name, center_point

    def trigger_servo(self):
        if self.log_func:
            self.log_func("Servo triggered: unlocking")
        self.unlock()
        sleep(0.1)
        if self.log_func:
            self.log_func("Servo triggered: locking")
        self.lock()

    def unlock(self):
        self.board.digital[self.pin].write(130)
        sleep(1)

    def lock(self):
        self.board.digital[self.pin].write(60)
        sleep(1)


class DetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Vision-Based Printing Defect Detection and Sorting System")

        self.title_label = tk.Label(root, text="Vision-Based Printing Defect Detection and Sorting System",
                                    font=("Courier", 22, "bold"), fg="#800000")
        self.title_label.pack(pady=10)

        self.fps_label = tk.Label(root, text="FPS: 0.00", font=("Helvetica", 12), fg="darkblue")
        self.fps_label.pack(pady=(0, 5))

        self.image_label = tk.Label(root)
        self.image_label.pack()

        button_frame = tk.Frame(root)
        button_frame.pack(pady=10)

        self.start_button = tk.Button(button_frame, text="Start Detection", command=self.start_video)
        self.start_button.grid(row=0, column=0, padx=5)

        self.stop_button = tk.Button(button_frame, text="Stop Detection", command=self.stop_video)
        self.stop_button.grid(row=0, column=1, padx=5)

        self.conf_label = tk.Label(button_frame, text="Confidence:")
        self.conf_label.grid(row=0, column=2, padx=(10, 0))

        self.conf_slider = ttk.Scale(button_frame, from_=0.1, to=1.0, orient='horizontal', value=0.5,
                                     command=self.update_confidence)
        self.conf_slider.grid(row=0, column=3)

        self.conf_value_label = tk.Label(button_frame, text="0.50")
        self.conf_value_label.grid(row=0, column=4, padx=(5, 0))

        self.status_label = tk.Label(root, text="Status: Idle", font=("Helvetica", 14), fg="blue")
        self.status_label.pack(pady=5)

        log_frame = tk.Frame(root)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0,10))

        self.log_text = tk.Text(log_frame, height=8, state=tk.DISABLED, bg="#f0f0f0")
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = tk.Scrollbar(log_frame, command=self.log_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.config(yscrollcommand=scrollbar.set)

        self.group_label = tk.Label(root, text="Developed by Group 7 (2008037-2008042)",
                                    font=("Helvetica", 14), fg="green")
        self.group_label.place(relx=0.5, rely=0.97, anchor='center')

        self.processor = VideoProcessor(log_func=self.log_message)
        self.cap = None
        self.running = False
        self.frame_image = None
        self.last_time = time()

    def log_message(self, message):
        timestamp = strftime("%H:%M:%S")
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

    def update_confidence(self, val):
        self.processor.confidence_threshold = float(val)
        self.conf_value_label.config(text=f"{float(val):.2f}")

    def start_video(self):
        if not self.running:
            self.cap = cv2.VideoCapture(0)
            self.running = True
            self.process_video()
            self.log_message("Detection started.")

    def stop_video(self):
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        black = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(black, "Webcam Stopped", (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        self.display_image(black)
        self.status_label.config(text="Status: Stopped", fg="gray")
        self.fps_label.config(text="FPS: 0.00")
        self.conf_value_label.config(text="0.00")
        self.conf_slider.set(0.5)
        self.log_message("Detection stopped.")

    def process_video(self):
        if self.running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                current_time = time()
                fps = 1 / (current_time - self.last_time)
                self.last_time = current_time
                self.fps_label.config(text=f"FPS: {fps:.2f}")

                processed, class_name, center_point = self.processor.process_frame(frame)

                height, _, _ = processed.shape
                line_y = int(height * 0.6)
                status = "Status: Detecting..."
                status_color = "blue"

                if class_name == "Miss_Print" and center_point is not None:
                    if line_y - 5 <= center_point[1] <= line_y + 5:
                        status = "Status: Miss_Print Detected!"
                        status_color = "red"
                    else:
                        status = "Status: Detecting..."
                        status_color = "blue"
                elif class_name is None:
                    status = "Status: No Fault Detected."
                    status_color = "green"

                self.status_label.config(text=status, fg=status_color)
                self.display_image(processed)

            self.root.after(15, self.process_video)

    def display_image(self, frame):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        imgtk = ImageTk.PhotoImage(Image.fromarray(img))
        self.frame_image = imgtk
        self.image_label.config(image=self.frame_image)


if __name__ == "__main__":
    root = tk.Tk()
    app = DetectionApp(root)
    root.mainloop()
    root.destroy()
