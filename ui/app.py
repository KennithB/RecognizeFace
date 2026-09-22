import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Optional
import cv2
import numpy as np
from PIL import Image, ImageTk

from core.camera import CameraStream
from core.engine import FaceRecognitionEngine
from core.profile_manager import ProfileManager


class FaceRecognitionApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Doorbell AI - Face Recognition System (InsightFace)")
        self.root.geometry("1240x820")
        self.root.minsize(1000, 680)

        # Style & theme
        self.bg_dark = "#14171a"
        self.panel_bg = "#212529"
        self.fg_text = "#f8f9fa"
        self.accent_color = "#00adb5"
        self.root.configure(bg=self.bg_dark)

        # Systems initialization
        self.profile_mgr = ProfileManager(profiles_dir="profiles")
        self.engine = FaceRecognitionEngine(model_name="buffalo_l", use_gpu=True)
        self.camera = CameraStream(source=0)

        # UI state
        self.is_streaming = False
        self.current_frame: Optional[np.ndarray] = None
        self.gallery_names, self.gallery_matrix = self.profile_mgr.get_gallery_matrix()
        self.prev_time = time.time()
        self.fps = 0.0

        self._build_ui()
        self._refresh_profile_list()

    def _build_ui(self):
        # Main Layout: 2 Columns (Left: Video, Right: Controls & Profiles)
        main_frame = tk.Frame(self.root, bg=self.bg_dark)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)

        # Left: Video Feed
        video_container = tk.Frame(main_frame, bg=self.panel_bg, relief=tk.FLAT, bd=2)
        video_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        # Title bar above video
        feed_header = tk.Frame(video_container, bg=self.panel_bg)
        feed_header.pack(fill=tk.X, padx=10, pady=8)

        self.lbl_feed_title = tk.Label(
            feed_header,
            text="● Live Surveillance Feed",
            font=("Segoe UI", 12, "bold"),
            bg=self.panel_bg,
            fg="#2ecc71",
        )
        self.lbl_feed_title.pack(side=tk.LEFT)

        self.lbl_fps = tk.Label(
            feed_header,
            text="FPS: 0.0",
            font=("Segoe UI", 10),
            bg=self.panel_bg,
            fg="#adb5bd",
        )
        self.lbl_fps.pack(side=tk.RIGHT)

        # Canvas for displaying camera frames
        self.canvas_video = tk.Label(
            video_container,
            text="Camera feed stopped.\nClick 'Start Camera' on the right panel.",
            font=("Segoe UI", 12),
            bg="#0d1117",
            fg="#8b949e",
        )
        self.canvas_video.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        # Status Bar beneath video
        self.lbl_status = tk.Label(
            video_container,
            text="Ready. Detected: 0 faces | Strangers: 0",
            font=("Segoe UI", 10),
            bg=self.panel_bg,
            fg="#e9ecef",
            anchor="w",
        )
        self.lbl_status.pack(fill=tk.X, padx=10, pady=(0, 8))

        # Right: Side Control Panel
        side_panel = tk.Frame(main_frame, bg=self.panel_bg, width=380)
        side_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 0))
        side_panel.pack_propagate(False)

        # --- Section 1: Camera Configuration ---
        cam_frame = tk.LabelFrame(
            side_panel,
            text=" Camera Hardware Connection ",
            font=("Segoe UI", 10, "bold"),
            bg=self.panel_bg,
            fg=self.fg_text,
            padx=10,
            pady=10,
        )
        cam_frame.pack(fill=tk.X, padx=10, pady=(10, 5))

        tk.Label(
            cam_frame,
            text="Source (Index e.g. 0, or RTSP URL):",
            bg=self.panel_bg,
            fg="#adb5bd",
            font=("Segoe UI", 9),
        ).pack(anchor="w")

        self.source_entry = ttk.Combobox(
            cam_frame,
            values=["0", "1", "2", "rtsp://192.168.1.100:554/stream1"],
            font=("Segoe UI", 10),
        )
        self.source_entry.set("0")
        self.source_entry.pack(fill=tk.X, pady=(2, 8))

        self.btn_camera = tk.Button(
            cam_frame,
            text="Start Camera",
            bg="#2ecc71",
            fg="#ffffff",
            font=("Segoe UI", 10, "bold"),
            relief=tk.FLAT,
            command=self._toggle_camera,
            cursor="hand2",
        )
        self.btn_camera.pack(fill=tk.X)

        # --- Section 2: Recognition Parameters ---
        param_frame = tk.LabelFrame(
            side_panel,
            text=" Detection Sensitivity ",
            font=("Segoe UI", 10, "bold"),
            bg=self.panel_bg,
            fg=self.fg_text,
            padx=10,
            pady=8,
        )
        param_frame.pack(fill=tk.X, padx=10, pady=5)

        tk.Label(
            param_frame,
            text="Cosine Similarity Threshold:",
            bg=self.panel_bg,
            fg="#adb5bd",
            font=("Segoe UI", 9),
        ).pack(anchor="w")

        self.threshold_var = tk.DoubleVar(value=0.45)
        self.threshold_slider = ttk.Scale(
            param_frame,
            from_=0.20,
            to=0.75,
            variable=self.threshold_var,
            orient=tk.HORIZONTAL,
            command=lambda v: self.lbl_threshold_val.config(text=f"{float(v):.2f}"),
        )
        self.threshold_slider.pack(fill=tk.X, pady=(2, 2))

        self.lbl_threshold_val = tk.Label(
            param_frame,
            text="0.45",
            bg=self.panel_bg,
            fg=self.accent_color,
            font=("Segoe UI", 9, "bold"),
        )
        self.lbl_threshold_val.pack(anchor="e")

        # --- Section 3: Profile Registration (Train Faces) ---
        train_frame = tk.LabelFrame(
            side_panel,
            text=" Train / Register Face Profile ",
            font=("Segoe UI", 10, "bold"),
            bg=self.panel_bg,
            fg=self.fg_text,
            padx=10,
            pady=10,
        )
        train_frame.pack(fill=tk.X, padx=10, pady=5)

        tk.Label(
            train_frame,
            text="Person Name:",
            bg=self.panel_bg,
            fg="#adb5bd",
            font=("Segoe UI", 9),
        ).pack(anchor="w")

        self.name_entry = tk.Entry(
            train_frame,
            font=("Segoe UI", 10),
            bg="#2c3036",
            fg="#ffffff",
            insertbackground="#ffffff",
        )
        self.name_entry.pack(fill=tk.X, pady=(2, 8))

        btn_row1 = tk.Frame(train_frame, bg=self.panel_bg)
        btn_row1.pack(fill=tk.X, pady=2)

        self.btn_capture = tk.Button(
            btn_row1,
            text="📸 Capture from Feed",
            bg="#3498db",
            fg="#ffffff",
            font=("Segoe UI", 9, "bold"),
            relief=tk.FLAT,
            command=self._register_from_camera,
            cursor="hand2",
        )
        self.btn_capture.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))

        self.btn_file = tk.Button(
            btn_row1,
            text="📁 From Image",
            bg="#6c757d",
            fg="#ffffff",
            font=("Segoe UI", 9, "bold"),
            relief=tk.FLAT,
            command=self._register_from_file,
            cursor="hand2",
        )
        self.btn_file.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(2, 0))

        # --- Section 4: Known Profiles List ---
        list_frame = tk.LabelFrame(
            side_panel,
            text=" Known Profiles Directory ",
            font=("Segoe UI", 10, "bold"),
            bg=self.panel_bg,
            fg=self.fg_text,
            padx=10,
            pady=10,
        )
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(5, 10))

        self.profile_listbox = tk.Listbox(
            list_frame,
            font=("Segoe UI", 10),
            bg="#2c3036",
            fg="#ffffff",
            selectbackground=self.accent_color,
            selectforeground="#000000",
            relief=tk.FLAT,
        )
        self.profile_listbox.pack(fill=tk.BOTH, expand=True, pady=(0, 8))

        self.btn_delete = tk.Button(
            list_frame,
            text="🗑 Delete Selected Profile",
            bg="#e74c3c",
            fg="#ffffff",
            font=("Segoe UI", 9, "bold"),
            relief=tk.FLAT,
            command=self._delete_profile,
            cursor="hand2",
        )
        self.btn_delete.pack(fill=tk.X)

    def _toggle_camera(self):
        if not self.is_streaming:
            source = self.source_entry.get().strip()
            self.camera.source = source
            if not self.camera.start():
                messagebox.showerror(
                    "Connection Error",
                    f"Could not open camera stream source '{source}'.\nCheck device connection or RTSP URL.",
                )
                return
            self.is_streaming = True
            self.btn_camera.config(text="Stop Camera", bg="#e74c3c")
            self.lbl_feed_title.config(text="● Live Surveillance Feed [ACTIVE]", fg="#2ecc71")
            self._update_loop()
        else:
            self.is_streaming = False
            self.camera.stop()
            self.btn_camera.config(text="Start Camera", bg="#2ecc71")
            self.lbl_feed_title.config(text="● Surveillance Feed [PAUSED]", fg="#e74c3c")
            self.canvas_video.config(image="", text="Camera stopped.")

    def _update_loop(self):
        if not self.is_streaming:
            return

        frame = self.camera.read_frame()
        if frame is not None:
            self.current_frame = frame.copy()

            # Run InsightFace analysis
            faces = self.engine.analyze_frame(frame)
            stranger_count = 0

            threshold = float(self.threshold_var.get())

            # Draw recognition overlays
            for face in faces:
                bbox = face.bbox.astype(int)
                x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]

                identity, score = self.engine.match_embedding(
                    face.normed_embedding,
                    self.gallery_names,
                    self.gallery_matrix,
                    threshold=threshold,
                )

                if identity == "STRANGER":
                    stranger_count += 1
                    color = (0, 0, 255)  # BGR: Red
                    label = f"STRANGER ({score:.2f})"
                else:
                    color = (0, 255, 0)  # BGR: Green
                    label = f"{identity} ({score:.2f})"

                # Draw bounding box & label tag
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Tag background
                tag_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(
                    frame,
                    (x1, y1 - tag_size[1] - 8),
                    (x1 + tag_size[0] + 6, y1),
                    color,
                    -1,
                )
                cv2.putText(
                    frame,
                    label,
                    (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255) if identity == "STRANGER" else (0, 0, 0),
                    2,
                )

            # Update FPS and Status
            now = time.time()
            dt = now - self.prev_time
            self.prev_time = now
            if dt > 0:
                self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt)
            self.lbl_fps.config(text=f"FPS: {self.fps:.1f}")

            status_text = f"Detected: {len(faces)} faces | Strangers: {stranger_count}"
            if stranger_count > 0:
                self.lbl_status.config(text=f"⚠️ ALERT: {status_text}", fg="#ff6b6b")
            else:
                self.lbl_status.config(text=f"Active: {status_text}", fg="#e9ecef")

            # Convert to PIL and Tk image
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = rgb_frame.shape[:2]

            # Fit to canvas dimensions maintaining aspect ratio
            cw = max(self.canvas_video.winfo_width(), 400)
            ch = max(self.canvas_video.winfo_height(), 300)
            scale = min(cw / w, ch / h)
            nw, nh = int(w * scale), int(h * scale)

            if nw > 0 and nh > 0:
                resized = cv2.resize(rgb_frame, (nw, nh), interpolation=cv2.INTER_AREA)
                img = ImageTk.PhotoImage(image=Image.fromarray(resized))
                self.canvas_video.img_ref = img
                self.canvas_video.config(image=img, text="")

        self.root.after(20, self._update_loop)

    def _register_from_camera(self):
        name = self.name_entry.get().strip()
        if not name:
            messagebox.showwarning("Input Required", "Please enter a person name before registering.")
            return

        if self.current_frame is None:
            messagebox.showwarning("No Frame", "Camera must be running to capture a face.")
            return

        emb = self.engine.extract_face_embedding(self.current_frame)
        if emb is None:
            messagebox.showerror(
                "Face Not Found",
                "No clear face detected in the current camera frame.\nPlease face the camera and try again.",
            )
            return

        self.profile_mgr.save_profile(name, emb)
        self._refresh_profile_list()
        self.name_entry.delete(0, tk.END)
        messagebox.showinfo("Success", f"Profile '{name}' registered successfully.")

    def _register_from_file(self):
        name = self.name_entry.get().strip()
        if not name:
            messagebox.showwarning("Input Required", "Please enter a person name before registering.")
            return

        file_path = filedialog.askopenfilename(
            title="Select Face Image",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.webp")],
        )
        if not file_path:
            return

        emb = self.engine.extract_from_file(file_path)
        if emb is None:
            messagebox.showerror("Face Not Found", "No face detected in selected image file.")
            return

        self.profile_mgr.save_profile(name, emb)
        self._refresh_profile_list()
        self.name_entry.delete(0, tk.END)
        messagebox.showinfo("Success", f"Profile '{name}' registered from image file.")

    def _delete_profile(self):
        selection = self.profile_listbox.curselection()
        if not selection:
            messagebox.showinfo("Selection Required", "Please select a profile to delete.")
            return

        name = self.profile_listbox.get(selection[0])
        if messagebox.askyesno("Confirm Deletion", f"Delete face profile '{name}'?"):
            self.profile_mgr.delete_profile(name)
            self._refresh_profile_list()

    def _refresh_profile_list(self):
        self.profile_listbox.delete(0, tk.END)
        profiles = self.profile_mgr.list_profiles()
        for p in profiles:
            self.profile_listbox.insert(tk.END, p)
        self.gallery_names, self.gallery_matrix = self.profile_mgr.get_gallery_matrix()

    def on_closing(self):
        self.is_streaming = False
        self.camera.stop()
        self.root.destroy()
