import numpy as np
import cv2
import tensorflow as tf

def preprocess_frame(frame):
    # Resize frame từ 1920x1080 về 640x640
    resized = cv2.resize(frame, (640, 640))

    # Chuyển BGR → RGB
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    # Normalize pixel từ [0, 255] về [0.0, 1.0]
    normalized = rgb / 255.0

    # Thêm chiều batch và chuyển sang float32
    input_data = np.expand_dims(normalized, axis=0).astype(np.float32)

    return input_data

# Load mô hình TFLite
interpreter = tf.lite.Interpreter(model_path="yolo11m_integer_quant.tflite")
interpreter.allocate_tensors()

# Lấy input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(input_details, output_details)

# # Load ảnh và tiền xử lý
# image = cv2.imread("the_loop.png")
# image_resized = cv2.resize(image, (640, 640))
# input_data = np.expand_dims(image_resized, axis=0).astype(np.float32) / 255.0

# # Inference
# interpreter.set_tensor(input_details[0]['index'], input_data)
# interpreter.invoke()

# # Lấy kết quả
# output_data = interpreter.get_tensor(output_details[0]['index'])

# # output_data thường có dạng [1, N, 6] (x1, y1, x2, y2, score, class)
# for det in output_data[0]:
#     x1, y1, x2, y2, conf, cls = det
#     if conf > 0.3:
#         print(f"Class {int(cls)}: {conf:.2f} at [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")

