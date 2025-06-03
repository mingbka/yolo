from ultralytics import YOLO
import cv2
import time
import numpy as np
import sys

class SpeedBuffer:
    def __init__(self, max_size=10):
        self.max_size = max_size
        self.buffer = []

    def add_speed(self, v):
        self.buffer.append(v)
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)

    def get_average(self):
        if not self.buffer:
            return 0.0
        return sum(self.buffer) / len(self.buffer)

class VehicleTracker:
    def __init__(self, model_path, tracker_config, class_names, class_colors, lane):
        self.model = YOLO(model_path)
        self.tracker_config = tracker_config
        self.class_names = class_names
        self.class_colors = class_colors
        self.lane = lane

        # Khởi tạo buffer lưu tối đa 10 tốc độ
        self.speed_buffer = SpeedBuffer(max_size=10)
    
    def draw_n_count(self, frame, boxes, track_ids, clss, total_count, vehicle_counter):
        line_y = 500

        cv2.line(frame, (0, line_y), (1920, line_y), (75, 176, 136), 2)

        for i, box in enumerate(boxes):
            track_id = track_ids[i] if i < len(track_ids) else -1
            cls_id = int(clss[i])

            class_name = self.class_names[cls_id]
            rgb = self.class_colors.get(cls_id, (255, 255, 255))  # fallback màu trắng
            color = (rgb[2], rgb[1], rgb[0])

            # Draw box 
            x1, y1, x2, y2 = map(int, box)

            if y2 < 250 or y1 > line_y:
                continue
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Counting
            if y2 > line_y and y1 < line_y:
                if track_id not in total_count:
                    total_count.append(track_id)
                    if class_name in vehicle_counter:
                        vehicle_counter[class_name] += 1

                else:
                    label = f"ID: {track_id} {class_name} counted"
                    (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(frame, (x1, y1 - label_height - baseline // 2), (x1 + label_width, y1), (0, 69, 255), -1)
                    cv2.putText(frame, label, (x1, y1 - baseline // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            else:
                label = f"ID:{track_id} {class_name}"
                (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x1, y1 - label_height - baseline // 2), (x1 + label_width, y1), color, -1)
                cv2.putText(frame, label, (x1, y1 - baseline // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    def cal_speed(self, boxes, clss, track_ids, car_speed):
        fps = 30
        car_length = 8.27
        trigger_line = 500

        for i, box in enumerate(boxes):
            cls_id = int(clss[i])
            track_id = track_ids[i] if i < len(track_ids) else -1
            x1, y1, x2, y2 = map(int, box)

            if cls_id != 1 or y2 < trigger_line :
                continue

            if track_id not in car_speed:
                car_speed[track_id] = {
                    "status": False,
                    "frame_count": 0,
                    "finished": False,
                    "speed": None
                    }

            info = car_speed[track_id]

            # Nếu đã tính vận tốc xong, bỏ qua
            if info["finished"]:
                continue

            # Bước 1: Bắt đầu tính tốc độ
            if y2 >= trigger_line and y2 < trigger_line + 15 and not info["status"]:
                info["status"] = True,
                info["frame_count"] = 1

            # Bước 2: Nếu đã bắt đầu
            elif info["status"]:
                info["frame_count"] += 1

                # Bước 3: Nếu xe đi qua vạch (end_y)
                if y1 >= trigger_line  and y1 < trigger_line + 15:
                    speed = (fps * car_length * 3.6) / info["frame_count"]
                    info["speed"] = speed
                    info["finished"] = True

                    self.speed_buffer.add_speed(speed)

        avg_speed = self.speed_buffer.get_average()
        return avg_speed

    def cal_pcu(self, frame, frame_count, lane, vehicle_pcu, vehicle_counter, avg_speed):
        fps = 30
        
        total_pcu = 0
        for vehicle_type in vehicle_counter:
            count = vehicle_counter.get(vehicle_type, 0)
            pcu = vehicle_pcu.get(vehicle_type, 0)
            total_pcu += count * pcu
        
        pcu_hour = total_pcu * (3600 * fps) / (frame_count * lane)

        if avg_speed is None or avg_speed <=0:
            pcu_km = 0
        else:
            pcu_km = pcu_hour / avg_speed
        
        text = f"PCU: {pcu_km:.2f} (PCU/km/lane)"

        x, y = 20, 50
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y - text_height - baseline), (x + text_width, y + baseline), (0, 0, 0), -1)
        alpha = 0.5
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.2, color=(255, 255, 255), thickness=2)
            
        return pcu_km

    def process_video(self, video_path):
        frame_count = 0

        total_count = []
        vehicle_counter = {
            'car': 0,
            'motor': 0,
            'truck': 0,
            'bus': 0
        }

        vehicle_pcu = {
            'car': 1,
            'motor': 0.3,
            'truck': 2,
            'bus': 2
        }

        car_speed = {}

        try:
            start_time = time.time()

            results = self.model.track(
                source=video_path,
                tracker=self.tracker_config,
                persist=True,
                conf=0.5,
                iou=0.6,
                show=False,
                verbose=False,
                stream=True,
                imgsz=512,
                vid_stride=1
            )

            for frame_idx, result in enumerate(results):
                frame = result.orig_img
                boxes = result.boxes.xyxy.cpu().numpy() if result.boxes.xyxy is not None else np.array([])
                track_ids = result.boxes.id.int().cpu().tolist() if result.boxes.id is not None else []
                # confs = result.boxes.conf.cpu().numpy() if result.boxes.conf is not None else np.array([])
                clss = result.boxes.cls.cpu().numpy() if result.boxes.cls is not None else np.array([])

                frame_count += 1

                self.draw_n_count(frame, boxes, track_ids, clss, total_count, vehicle_counter)
                avg_speed = self.cal_speed(boxes, clss, track_ids, car_speed)
                pcu_km = self.cal_pcu(frame, frame_count, self.lane, vehicle_pcu, vehicle_counter, avg_speed)

                cv2.imshow("YOLO Tracking", frame)               

                key = cv2.waitKey(1) & 0xFF 
                if key == 27:
                    break
                if key ==13:
                    while True:
                        cv2.waitKey(0) & 0xFF
                        break

            end_time = time.time()
            cv2.destroyAllWindows()

        except Exception as e:
            print(f"Đã xảy ra lỗi trong quá trình tracking: {e}")
            import traceback
            traceback.print_exc()

        if frame_count > 0:
            avg_fps = frame_count / (end_time - start_time)
            print(f"Tốc độ xử lý trung bình (FPS): {avg_fps:.2f}")
            print(f"PCU: {pcu_km:.2f} (PCU/km/lane)")
        else:
            print("Không có khung hình nào được xử lý.")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Cách dùng: python track_pytorch.py <model_path> <video_path> <lane>")
        sys.exit(1)

    model_path = sys.argv[1]
    video_path = sys.argv[2]
    lane = int(sys.argv[3])

    # Cấu hình
    TRACKER_CONFIG_PATH = 'bytetrack.yaml'

    # Định nghĩa class
    classNames = ["bus", "car", "motor", "truck"]

    classColors = {
        0: (255, 218, 185), # peach (bus)
        1: (137, 207, 240), # baby blue (car)
        2: (152, 255, 152), # mint (motor)
        3: (88, 219, 255), # mustard (truck)
    }

    processor = VehicleTracker(model_path, TRACKER_CONFIG_PATH, classNames, classColors, lane)
    processor.process_video(video_path)
