from ultralytics import YOLO
import cv2
import time
 
cap = cv2.VideoCapture("video4.mp4")  
if not cap.isOpened():
    print("Lỗi: Không thể mở file video.")
    exit()
 
# Model YOLO
try:
    model = YOLO("yolo_weights.pt", task='detect')
    
except Exception as e:
    print(f"Lỗi khi tải model YOLO: {e}")
    exit()

classNames = [" ", "car", "motor", "truck", "bus"]
 
# --- Khởi tạo biến đếm thời gian ---
total_processing_time = 0.0
frame_count = 0
# ------------------------------------
 
while True:
    success, img = cap.read()
    if not success:
        print("Kết thúc video.")
        break
    start_time = time.time()
    results = model(img, stream=True, verbose=False)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Confidence
            conf = box.conf[0]
            # Class Name
            cls = int(box.cls[0]) 
            cls_name = classNames[cls]

            display_text = f"{cls_name} {conf:.2f}"

            if (conf > 0.5):
                cv2.rectangle(img, (x1, y1), (x2, y2), (100, 235, 120), thickness=2)
                cv2.putText(img, display_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 235, 120), 2)
            
 
    end_time = time.time()
    frame_processing_time = end_time - start_time
    total_processing_time += frame_processing_time
    frame_count += 1
 
    cv2.imshow("Image", img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

if frame_count > 0:
    average_time_per_frame = total_processing_time / frame_count
    average_fps = 1.0 / average_time_per_frame
    print(f"Tốc độ xử lý trung bình (FPS): {average_fps:.2f}")