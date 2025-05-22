from ultralytics import YOLO
import cv2
import time
import numpy as np
import sys

class VideoProcessor:
    def __init__(self, model_path, tracker_config, class_names, class_colors):
        self.model = YOLO(model_path)
        self.tracker_config = tracker_config
        self.class_names = class_names
        self.class_colors = class_colors
    
    def _draw_boxes(self, frame, boxes, track_ids, confs, clss):
        for i, box in enumerate(boxes):
            track_id = track_ids[i] if i < len(track_ids) else -1
            conf = confs[i]
            cls_id = int(clss[i])

            class_name = self.class_names[cls_id - 1]
            rgb = self.class_colors.get(cls_id - 1, (255, 255, 255))  # fallback màu trắng
            color = (rgb[2], rgb[1], rgb[0])

            x1, y1, x2, y2 = map(int, box)
            label = f"ID:{track_id} {class_name} {conf:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - label_height - baseline), (x1 + label_width, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - baseline // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    def process_video(self, video_path):
        frame_count = 0

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
                stream=True
            )

            for frame_idx, result in enumerate(results):
                frame = result.orig_img
                boxes = result.boxes.xyxy.cpu().numpy() if result.boxes.xyxy is not None else np.array([])
                track_ids = result.boxes.id.int().cpu().tolist() if result.boxes.id is not None else []
                confs = result.boxes.conf.cpu().numpy() if result.boxes.conf is not None else np.array([])
                clss = result.boxes.cls.cpu().numpy() if result.boxes.cls is not None else np.array([])

                self._draw_boxes(frame, boxes, track_ids, confs, clss)

                cv2.imshow("YOLO Tracking", frame)

                frame_count += 1

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
            avg_time = (end_time - start_time) / frame_count
            avg_fps = 1.0 / avg_time
            print(f"Tốc độ xử lý trung bình (FPS): {avg_fps:.2f}")
        else:
            print("Không có khung hình nào được xử lý.")

# print(f"Tổng số xe hơi đếm được: {totalCarCount}")
# print(f"Tổng số xe máy đếm được: {totalBikeCount}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Cách dùng: python track_pytorch.py <model_path> <video_path>")
        sys.exit(1)

    model_path = sys.argv[1]
    video_path = sys.argv[2]

    # Cấu hình
    TRACKER_CONFIG_PATH = 'bytetrack.yaml'
    OUTPUT_PATH = "../yolo/output_video/output_track-13-5(1).mp4"

    # Khởi tạo video writer
    # fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    # fps    = 30
    # w      = 1920
    # h      = 1080
    # writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (w, h))

    # Định nghĩa class
    classNames = ["car", "motor", "truck", "bus"]

    classColors = {
        0: (137, 207, 240), # baby blue (car)
        1: (152, 255, 152), # mint (motor)
        2: (230, 230, 250), # lavender (truck)
        -1: (255, 218, 185), # peach (bus)
    }

    # limits = [550, 650, 873, 550]
    # totalCount = []
    # totalBikeCount = 0
    # totalCarCount = 0
    # totalTruckCount = 0
    # totalBusCount = 0

    processor = VideoProcessor(model_path, TRACKER_CONFIG_PATH, classNames, classColors)
    processor.process_video(video_path)
