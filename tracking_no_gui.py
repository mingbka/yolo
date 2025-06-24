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
    
class EMAFilter:
    def __init__(self, alpha):
        self.alpha = alpha
        self.ema = None
    
    def update(self, total_pcu):
        if self.ema is None:
            self.ema = total_pcu
        else:
            self.ema = self.alpha * total_pcu + (1 - self.alpha) * self.ema
        return self.ema

class VehicleTracker:
    def __init__(self, model_path, tracker_config, class_names, lane):
        self.model = YOLO(model_path)
        self.tracker_config = tracker_config
        self.class_names = class_names
        self.lane = lane
        self.ema_value = 0
        self.vid_stride = 2

        # Khởi tạo buffer lưu tối đa 10 tốc độ
        self.speed_buffer = SpeedBuffer(max_size=10)

        # Hàm tính EMA
        self.ema_filter = EMAFilter(alpha=2/(10+1))
    
    def count(self, boxes, track_ids, clss, total_count, vehicle_counter):
        line_y = 500

        for i, box in enumerate(boxes):
            track_id = track_ids[i] if i < len(track_ids) else -1
            cls_id = int(clss[i])
            class_name = self.class_names[cls_id]
            x1, y1, x2, y2 = map(int, box)

            # Counting
            if y2 > line_y and y1 < line_y:
                if track_id not in total_count:
                    total_count.append(track_id)
                    if class_name in vehicle_counter:
                        vehicle_counter[class_name] += 1

    def cal_speed(self, boxes, clss, track_ids, car_speed):
        fps = 30 / self.vid_stride
        meter_length = 6
        pixel_length = 180
        trigger_line = 430

        for i, box in enumerate(boxes):
            cls_id = int(clss[i])
            x1, y1, x2, y2 = map(int, box)

            if cls_id != 1 or y2 < trigger_line :
                continue

            track_id = track_ids[i] if i < len(track_ids) else -1
            
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
            if y2 >= trigger_line and y2 <= trigger_line + 25 and not info["status"]:
                info["status"] = True
                info["frame_count"] = 1

            # Bước 2: Nếu đã bắt đầu
            elif info["status"]:
                info["frame_count"] += 1

                # Bước 3: Nếu xe đi qua vạch (end_y)
                if (y2 >= trigger_line + pixel_length):
                    speed = (fps * meter_length * 3.6) / info["frame_count"]
                    info["speed"] = speed
                    info["finished"] = True

                    self.speed_buffer.add_speed(speed)

        avg_speed = self.speed_buffer.get_average()        
        return avg_speed

    def cal_ema(self, vehicle_pcu, vehicle_counter):
        total_pcu = 0

        # B1: Calculate sum of PCU
        for vehicle_type in vehicle_counter:
            count = vehicle_counter.get(vehicle_type, 0)
            pcu = vehicle_pcu.get(vehicle_type, 0)
            total_pcu += count * pcu
            print(f"{vehicle_type}: {count}")

        # B2: Reset vehicle_counter 
            vehicle_counter[vehicle_type] = 0

        # B3: Calculate EMA
        ema_value = self.ema_filter.update(total_pcu)

        return ema_value
            
    def cal_density(self, ema_value, avg_speed):
        density = ema_value * 60 / (avg_speed * self.lane) if avg_speed != 0 else 0
        print(f"Density: {density:.1f}")

        if density < 100:
            print("Status: Moderate traffic")
        elif 120 < density < 200:
            print("Status: Heavy traffic")
        else:
            print("Status: Congested traffic")

    def process_video(self, video_path):
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
            'bus': 2.5
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
                vid_stride=self.vid_stride
            )

            for frame_count, result in enumerate(results):
                boxes = result.boxes.xyxy.cpu().numpy() if result.boxes.xyxy is not None else np.array([])
                track_ids = result.boxes.id.int().cpu().tolist() if result.boxes.id is not None else []
                clss = result.boxes.cls.cpu().numpy() if result.boxes.cls is not None else np.array([])

                self.count(boxes, track_ids, clss, total_count, vehicle_counter)
                avg_speed = self.cal_speed(boxes, clss, track_ids, car_speed)

                if frame_count % (1800 / self.vid_stride) == 0 and frame_count > 0:
                    self.ema_value = self.cal_ema(vehicle_pcu, vehicle_counter)
                    self.cal_density(self.ema_value, avg_speed)            

            end_time = time.time()
            cv2.destroyAllWindows()

        except Exception as e:
            print(f"Đã xảy ra lỗi trong quá trình tracking: {e}")
            import traceback
            traceback.print_exc()

        if frame_count > 0:
            avg_fps = frame_count * self.vid_stride / (end_time - start_time)
            print(f"Tốc độ xử lý trung bình (FPS): {avg_fps:.2f}")
        else:
            print("Không có khung hình nào được xử lý.")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Syntax: python track_pytorch.py <model_path> <video_path> <lane>")
        sys.exit(1)

    model_path = sys.argv[1]
    video_path = sys.argv[2]
    lane = int(sys.argv[3])

    # Cấu hình
    TRACKER_CONFIG_PATH = 'bytetrack.yaml'

    # Định nghĩa class
    classNames = ["bus", "car", "motor", "truck"]

    processor = VehicleTracker(model_path, TRACKER_CONFIG_PATH, classNames, lane)
    processor.process_video(video_path)

