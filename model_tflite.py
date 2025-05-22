import numpy as np
import cv2
import tensorflow as tf
import sys

class VehicleTracking:
    def __init__(self, model_path: str, video_path):
        self.model_path = model_path

        self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise IOError("Lỗi: Không thể mở file video.")
        
        self.classNames = ["car", "motor", "truck", "bus"]

    def preprocess(self, frame):
        h, w = frame.shape[:2]
        pad = (w - h) // 2
        padded = cv2.copyMakeBorder(frame, pad, pad, 0, 0, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        resized = cv2.resize(padded, (640, 640))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        return np.expand_dims(rgb / 255.0, axis=0).astype(np.float32)
    
    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Kết thúc video")
                break

            input_data = self.preprocess(frame)

            # Inference
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()

            # Lấy kết quả
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])

            # output_data thường có dạng [1, N, 6] (x1, y1, x2, y2, score, class)
            for det in output_data[0]:
                x1, y1, x2, y2, conf, cls = det
                cls_name = self.classNames[int(cls)]
                if conf > 0.5:
                    print(f"{cls_name}: {conf:.2f} at [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Cách dùng: python model_tflite.py <model_path> <video_path>")
        sys.exit(1)

    model_path = sys.argv[1]
    video_path = sys.argv[2]

    processor = VehicleTracking(model_path, video_path)
    processor.run()

