import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import cv2
import os
from tracking_no_gui import VehicleTracker
import time

class VehicleTrackingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Vehicle Tracking System")
        self.root.geometry("600x700")
        self.root.resizable(True, True)
        
        # Biến điều khiển
        self.tracking_thread = None
        self.is_running = False
        self.stop_tracking = False
        
        # Thiết lập giao diện
        self.setup_ui()
        
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, text="Traffic Monitoring System", 
                               font=("Arial", 18, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 30))
        
        # Input Section
        input_frame = ttk.LabelFrame(main_frame, text="Cấu hình", padding="15")
        input_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 20))
        
        # Video Source
        ttk.Label(input_frame, text="Nguồn video:").grid(row=1, column=0, sticky=tk.W, pady=5)
        
        # Radio buttons for video source
        self.source_var = tk.StringVar(value="file")
        source_frame = ttk.Frame(input_frame)
        source_frame.grid(row=1, column=1, columnspan=2, sticky=tk.W, padx=10, pady=5)
        
        ttk.Radiobutton(source_frame, text="File video", variable=self.source_var, 
                       value="file", command=self.toggle_video_input).pack(side=tk.LEFT)
        ttk.Radiobutton(source_frame, text="Camera", variable=self.source_var, 
                       value="camera", command=self.toggle_video_input).pack(side=tk.LEFT, padx=(20, 0))
        
        # Video Path
        ttk.Label(input_frame, text="Đường dẫn video:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.video_path_var = tk.StringVar()
        self.video_entry = ttk.Entry(input_frame, textvariable=self.video_path_var, width=40)
        self.video_entry.grid(row=2, column=1, padx=(10, 5), pady=5)
        self.video_browse_btn = ttk.Button(input_frame, text="Browse", 
                                          command=self.browse_video)
        self.video_browse_btn.grid(row=2, column=2, padx=5, pady=5)
        
        # Camera ID (hidden by default)
        self.camera_label = ttk.Label(input_frame, text="Camera")
        
        # Number of lanes
        ttk.Label(input_frame, text="Số làn đường:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.lanes_var = tk.StringVar(value="4")
        lanes_spinbox = ttk.Spinbox(input_frame, from_=1, to=10, textvariable=self.lanes_var, width=10)
        lanes_spinbox.grid(row=3, column=1, sticky=tk.W, padx=10, pady=5)
        
        # Control Buttons
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=2, column=0, columnspan=3, pady=20)
        
        self.start_btn = ttk.Button(control_frame, text="Bắt đầu Tracking", 
                                   command=self.start_tracking, style="Accent.TButton")
        self.start_btn.pack(side=tk.LEFT, padx=10)
        
        self.stop_btn = ttk.Button(control_frame, text="Dừng Tracking", 
                                  command=self.stop_tracking_func, state="disabled")
        self.stop_btn.pack(side=tk.LEFT, padx=10)
        
        # Status Section
        status_frame = ttk.LabelFrame(main_frame, text="Trạng thái", padding="15")
        status_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 20))
        
        self.status_var = tk.StringVar(value="Sẵn sàng")
        status_label = ttk.Label(status_frame, textvariable=self.status_var, font=("Arial", 10))
        status_label.pack()
        
        # Progress bar
        self.progress = ttk.Progressbar(status_frame, mode='indeterminate')
        self.progress.pack(fill=tk.X, pady=(10, 0))
        
        # Log Text Area
        log_frame = ttk.LabelFrame(main_frame, text="Log", padding="10")
        log_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        # Text widget with scrollbar
        text_frame = ttk.Frame(log_frame)
        text_frame.pack(fill=tk.BOTH, expand=True)
        
        self.log_text = tk.Text(text_frame, height=8, wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(4, weight=1)
        
    def toggle_video_input(self):
        """Toggle between file and camera input"""
        if self.source_var.get() == "camera":
            # Hide video file input, show camera input
            self.video_entry.grid_remove()
            self.video_browse_btn.grid_remove()
            self.camera_label.grid(row=2, column=1, sticky=tk.W, padx=10, pady=5)
        else:
            # Show video file input, hide camera input
            self.camera_label.grid_remove()
            self.video_entry.grid(row=2, column=1, padx=(10, 5), pady=5)
            self.video_browse_btn.grid(row=2, column=2, padx=5, pady=5)
    
    def browse_video(self):
        """Browse for video file"""
        filename = filedialog.askopenfilename(
            title="Chọn file video",
            filetypes=[("Video files", "*.mp4"), ("All files", "*.*")]
        )
        if filename:
            self.video_path_var.set(filename)
    
    def log_message(self, message):
        """Add message to log"""
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()
        
        # Check video source
        if self.source_var.get() == "file":
            video_path = self.video_path_var.get().strip()
            if not video_path or not os.path.exists(video_path):
                messagebox.showerror("Lỗi", "Vui lòng chọn file video hợp lệ!")
                return False
        
        # Check number of lanes
        try:
            lanes = int(self.lanes_var.get())
            if lanes < 1:
                raise ValueError
        except ValueError:
            messagebox.showerror("Lỗi", "Số làn đường phải là số nguyên dương!")
            return False
        
        return True
    
    def start_tracking(self):
        """Start the tracking process"""      
        if self.is_running:
            messagebox.showwarning("Cảnh báo", "Chương trình đang chạy!")
            return
        
        # Update UI
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.status_var.set("Đang khởi tạo...")
        self.progress.start()
        self.log_text.delete(1.0, tk.END)
        
        # Start tracking in separate thread
        self.stop_tracking = False
        self.tracking_thread = threading.Thread(target=self.run_tracking)
        self.tracking_thread.daemon = True
        self.tracking_thread.start()
    
    def run_tracking(self):
        """Run the tracking process"""
        try:
            self.is_running = True
            
            # Get parameters
            model_path = 'weights/yv8n24.pt'
            lanes = int(self.lanes_var.get())
            
            if self.source_var.get() == "camera":
                video_source = int(self.camera_var.get())
            else:
                video_source = self.video_path_var.get().strip()
            
            self.log_message(f"Nguồn video: {video_source}")
            self.log_message(f"Số làn đường: {lanes}")
            
            # Update status
            self.root.after(0, lambda: self.status_var.set("Đang chạy tracking..."))
            
            # Configuration
            TRACKER_CONFIG_PATH = 'bytetrack.yaml'
            
            # Class definitions
            classNames = ["bus", "car", "motor", "truck"]
            
            # Create tracker
            processor = VehicleTracker(model_path, TRACKER_CONFIG_PATH, classNames, lanes)
            
            # Modify the process_video method to check for stop signal
            self.run_tracking_with_stop_check(processor, video_source)
            
        except Exception as e:
            self.log_message(f"Lỗi: {str(e)}")
            messagebox.showerror("Lỗi", f"Đã xảy ra lỗi: {str(e)}")
        finally:
            self.is_running = False
            self.root.after(0, self.tracking_finished)
    
    def run_tracking_with_stop_check(self, processor, video_source):
        """Modified tracking with stop check"""
        vehicle_counter = {'car': 0, 'motor': 0, 'truck': 0, 'bus': 0}
        vehicle_pcu = {'car': 1, 'motor': 0.3, 'truck': 2, 'bus': 2.5}
        car_speed = {}
        
        try:
            start_time = time.time()
            
            results = processor.model.track(
                source=video_source,
                tracker=processor.tracker_config,
                persist=True,
                conf=0.5,
                iou=0.3,
                show=False,
                verbose=False,
                stream=True,
                imgsz=512,
                vid_stride=processor.vid_stride
            )
            
            for frame_count, result in enumerate(results):
                if self.stop_tracking:
                    self.log_message("Đã dừng tracking")
                    end_time = time.time()
                    avg_fps = frame_count / (end_time - start_time)
                    self.log_message(f"FPS: {avg_fps:.1f}")
                    break
                
                frame = result.orig_img
                boxes = result.boxes.xyxy.cpu().numpy() if result.boxes.xyxy is not None else []
                track_ids = result.boxes.id.int().cpu().tolist() if result.boxes.id is not None else []
                clss = result.boxes.cls.cpu().numpy() if result.boxes.cls is not None else []
                
                processor.count(boxes, track_ids, clss, vehicle_counter)
                avg_speed = processor.cal_speed(boxes, clss, track_ids, car_speed)
                
                if frame_count % (1800 / processor.vid_stride) == 0 and frame_count > 0:
                    processor.ema_value = processor.cal_ema(vehicle_pcu, vehicle_counter)
                    processor.cal_density(processor.ema_value, avg_speed)
            
            cv2.destroyAllWindows()
            
        except Exception as e:
            self.log_message(f"Lỗi trong quá trình tracking: {str(e)}")
            raise
    
    def stop_tracking_func(self):
        """Stop the tracking process"""
        if self.is_running:
            self.stop_tracking = True
            self.log_message("Đang dừng tracking...")
            self.status_var.set("Đang dừng...")
        
    def tracking_finished(self):
        """Called when tracking is finished"""
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.status_var.set("Hoàn thành")
        self.progress.stop()
        cv2.destroyAllWindows()

def main():
    # Create main window
    root = tk.Tk()
    
    # Set style
    style = ttk.Style()
    style.theme_use('clam')
    
    # Create app
    app = VehicleTrackingApp(root)
    
    # Start GUI
    root.mainloop()

if __name__ == "__main__":
    main()



