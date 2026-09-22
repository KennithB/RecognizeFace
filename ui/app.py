import time
from tkinter import filedialog, messagebox
from typing import Optional
import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image, ImageTk

from core.camera import CameraStream
from core.engine import FaceRecognitionEngine
from core.profile_manager import ProfileManager


class FaceRecognitionApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Window configuration
        self.title("Face Recognition System")
        self.geometry("1200x780")
        self.minsize(980, 640)
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # System components
        self.profile_mgr = ProfileManager(profiles_dir="profiles")
        self.engine = FaceRecognitionEngine(model_name="buffalo_l", use_gpu=True)
        self.camera = CameraStream(source=0)

        # Runtime state
        self.is_streaming = False
        self.current_frame: Optional[np.ndarray] = None
        self.gallery_names, self.gallery_matrix = self.profile_mgr.get_gallery_matrix()
        self.prev_time = time.time()
        self.fps = 0.0

        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        self._build_layout()
        self._refresh_profiles()

    def _build_layout(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=0)
        self.grid_rowconfigure(0, weight=1)

        # ================= Video Container (Left) =================
        self.video_panel = ctk.CTkFrame(self, corner_radius=8)
        self.video_panel.grid(row=0, column=0, sticky="nsew", padx=(16, 8), pady=16)
        self.video_panel.grid_rowconfigure(1, weight=1)
        self.video_panel.grid_columnconfigure(0, weight=1)

        # Video Header
        self.header_frame = ctk.CTkFrame(self.video_panel, fg_color="transparent")
        self.header_frame.grid(row=0, column=0, sticky="ew", padx=16, pady=(14, 8))
        self.header_frame.grid_columnconfigure(0, weight=1)

        self.lbl_status = ctk.CTkLabel(
            self.header_frame,
            text="Camera Idle",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#8a94a6",
        )
        self.lbl_status.grid(row=0, column=0, sticky="w")

        self.lbl_fps = ctk.CTkLabel(
            self.header_frame,
            text="0.0 FPS",
            font=ctk.CTkFont(size=13),
            text_color="#8a94a6",
        )
        self.lbl_fps.grid(row=0, column=1, sticky="e")

        # Video Viewport
        self.video_display = ctk.CTkLabel(
            self.video_panel,
            text="Feed stopped\nStart camera to begin detection",
            font=ctk.CTkFont(size=14),
            fg_color="#121316",
            corner_radius=6,
        )
        self.video_display.grid(row=1, column=0, sticky="nsew", padx=16, pady=(0, 16))

        # ================= Controls Sidebar (Right) =================
        self.sidebar = ctk.CTkScrollableFrame(self, width=320, corner_radius=8)
        self.sidebar.grid(row=0, column=1, sticky="nsew", padx=(8, 16), pady=16)

        # --- Section: Camera ---
        lbl_cam = ctk.CTkLabel(
            self.sidebar,
            text="Camera Source",
            font=ctk.CTkFont(size=13, weight="bold"),
            anchor="w",
        )
        lbl_cam.pack(fill="x", pady=(8, 4))

        self.entry_source = ctk.CTkEntry(
            self.sidebar,
            placeholder_text="0, 1 or RTSP URL",
            height=34,
        )
        self.entry_source.insert(0, "0")
        self.entry_source.pack(fill="x", pady=(0, 8))

        self.btn_camera = ctk.CTkButton(
            self.sidebar,
            text="Start Camera",
            height=36,
            command=self._toggle_camera,
        )
        self.btn_camera.pack(fill="x", pady=(0, 16))

        # --- Section: Sensitivity ---
        lbl_thresh = ctk.CTkLabel(
            self.sidebar,
            text="Matching Threshold",
            font=ctk.CTkFont(size=13, weight="bold"),
            anchor="w",
        )
        lbl_thresh.pack(fill="x", pady=(4, 2))

        self.lbl_thresh_val = ctk.CTkLabel(
            self.sidebar,
            text="0.45",
            font=ctk.CTkFont(size=12),
            text_color="#8a94a6",
            anchor="w",
        )
        self.lbl_thresh_val.pack(fill="x", pady=(0, 2))

        self.slider_thresh = ctk.CTkSlider(
            self.sidebar,
            from_=0.20,
            to=0.75,
            number_of_steps=55,
            command=self._on_threshold_change,
        )
        self.slider_thresh.set(0.45)
        self.slider_thresh.pack(fill="x", pady=(0, 16))

        # --- Section: Profile Registration ---
        lbl_enroll = ctk.CTkLabel(
            self.sidebar,
            text="Enroll Face Profile",
            font=ctk.CTkFont(size=13, weight="bold"),
            anchor="w",
        )
        lbl_enroll.pack(fill="x", pady=(4, 4))

        self.entry_name = ctk.CTkEntry(
            self.sidebar,
            placeholder_text="Person Name",
            height=34,
        )
        self.entry_name.pack(fill="x", pady=(0, 8))

        self.btn_capture = ctk.CTkButton(
            self.sidebar,
            text="Capture from Stream",
            height=34,
            fg_color="#2b313a",
            hover_color="#363d48",
            command=self._register_from_stream,
        )
        self.btn_capture.pack(fill="x", pady=(0, 6))

        self.btn_import = ctk.CTkButton(
            self.sidebar,
            text="Import Photo File",
            height=34,
            fg_color="#2b313a",
            hover_color="#363d48",
            command=self._register_from_file,
        )
        self.btn_import.pack(fill="x", pady=(0, 16))

        # --- Section: Registered Profiles ---
        lbl_profiles = ctk.CTkLabel(
            self.sidebar,
            text="Registered Profiles",
            font=ctk.CTkFont(size=13, weight="bold"),
            anchor="w",
        )
        lbl_profiles.pack(fill="x", pady=(4, 4))

        self.profiles_container = ctk.CTkFrame(self.sidebar, fg_color="#181a1f", corner_radius=6)
        self.profiles_container.pack(fill="x", pady=(0, 8))

        self.selected_profile: Optional[str] = None
        self.profile_buttons = []

        self.btn_delete = ctk.CTkButton(
            self.sidebar,
            text="Delete Profile",
            height=34,
            fg_color="#992d22",
            hover_color="#b83528",
            command=self._delete_selected_profile,
        )
        self.btn_delete.pack(fill="x", pady=(0, 12))

    def _on_threshold_change(self, val):
        self.lbl_thresh_val.configure(text=f"{float(val):.2f}")

    def _toggle_camera(self):
        if not self.is_streaming:
            source = self.entry_source.get().strip()
            self.camera.source = source
            if not self.camera.start():
                messagebox.showerror("Error", f"Failed to open camera: '{source}'")
                return
            self.is_streaming = True
            self.btn_camera.configure(text="Stop Camera", fg_color="#992d22", hover_color="#b83528")
            self.lbl_status.configure(text="Active Monitoring", text_color="#2ecc71")
            self._update_loop()
        else:
            self.is_streaming = False
            self.camera.stop()
            self.btn_camera.configure(text="Start Camera", fg_color=["#3B8ED0", "#1F6AA5"], hover_color=["#36719F", "#144870"])
            self.lbl_status.configure(text="Camera Stopped", text_color="#8a94a6")
            self.lbl_fps.configure(text="0.0 FPS")
            self.video_display.configure(image="", text="Feed stopped\nStart camera to begin detection")

    def _update_loop(self):
        if not self.is_streaming:
            return

        frame = self.camera.read_frame()
        if frame is not None:
            self.current_frame = frame.copy()
            faces = self.engine.analyze_frame(frame)
            threshold = float(self.slider_thresh.get())
            stranger_count = 0

            for face in faces:
                x1, y1, x2, y2 = face.bbox.astype(int)
                name, score = self.engine.match_embedding(
                    face.normed_embedding,
                    self.gallery_names,
                    self.gallery_matrix,
                    threshold=threshold,
                )

                if name == "STRANGER":
                    stranger_count += 1
                    box_color = (0, 0, 255)
                    label_text = f"STRANGER ({score:.2f})"
                else:
                    box_color = (0, 230, 115)
                    label_text = f"{name} ({score:.2f})"

                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), box_color, -1)
                cv2.putText(
                    frame,
                    label_text,
                    (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255) if name == "STRANGER" else (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )

            # Update FPS & Status
            now = time.time()
            dt = now - self.prev_time
            self.prev_time = now
            if dt > 0:
                self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt)
            self.lbl_fps.configure(text=f"{self.fps:.1f} FPS")

            status_msg = f"{len(faces)} Face(s) Detected"
            if stranger_count > 0:
                status_msg += f" — {stranger_count} Stranger(s)"
                self.lbl_status.configure(text=status_msg, text_color="#ff5252")
            else:
                self.lbl_status.configure(text=status_msg, text_color="#2ecc71")

            # Scale and display image
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]

            disp_w = max(self.video_display.winfo_width(), 480)
            disp_h = max(self.video_display.winfo_height(), 320)
            scale = min(disp_w / w, disp_h / h)
            nw, nh = max(1, int(w * scale)), max(1, int(h * scale))

            resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
            img = ImageTk.PhotoImage(image=Image.fromarray(resized))
            self.video_display.img_ref = img
            self.video_display.configure(image=img, text="")

        self.after(20, self._update_loop)

    def _register_from_stream(self):
        name = self.entry_name.get().strip()
        if not name:
            messagebox.showwarning("Warning", "Enter a person name first.")
            return

        if self.current_frame is None:
            messagebox.showwarning("Warning", "Camera is not streaming.")
            return

        emb = self.engine.extract_face_embedding(self.current_frame)
        if emb is None:
            messagebox.showerror("Error", "No face detected in current frame.")
            return

        self.profile_mgr.save_profile(name, emb)
        self.entry_name.delete(0, "end")
        self._refresh_profiles()
        messagebox.showinfo("Success", f"Profile '{name}' registered.")

    def _register_from_file(self):
        name = self.entry_name.get().strip()
        if not name:
            messagebox.showwarning("Warning", "Enter a person name first.")
            return

        path = filedialog.askopenfilename(
            title="Select Face Photo",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.webp *.bmp")],
        )
        if not path:
            return

        emb = self.engine.extract_from_file(path)
        if emb is None:
            messagebox.showerror("Error", "No face found in selected image.")
            return

        self.profile_mgr.save_profile(name, emb)
        self.entry_name.delete(0, "end")
        self._refresh_profiles()
        messagebox.showinfo("Success", f"Profile '{name}' registered.")

    def _select_profile(self, name: str):
        self.selected_profile = name
        for btn, p_name in self.profile_buttons:
            if p_name == name:
                btn.configure(fg_color="#3a4454")
            else:
                btn.configure(fg_color="transparent")

    def _delete_selected_profile(self):
        if not self.selected_profile:
            messagebox.showinfo("Notice", "Select a profile to delete.")
            return

        if messagebox.askyesno("Confirm", f"Delete profile '{self.selected_profile}'?"):
            self.profile_mgr.delete_profile(self.selected_profile)
            self.selected_profile = None
            self._refresh_profiles()

    def _refresh_profiles(self):
        for btn, _ in self.profile_buttons:
            btn.destroy()
        self.profile_buttons.clear()

        profiles = self.profile_mgr.list_profiles()
        if not profiles:
            lbl_empty = ctk.CTkLabel(
                self.profiles_container,
                text="No profiles registered",
                font=ctk.CTkFont(size=12),
                text_color="#6c757d",
            )
            lbl_empty.pack(pady=10)
            self.profile_buttons.append((lbl_empty, ""))
        else:
            for name in profiles:
                btn = ctk.CTkButton(
                    self.profiles_container,
                    text=name,
                    font=ctk.CTkFont(size=12),
                    anchor="w",
                    height=30,
                    fg_color="transparent",
                    hover_color="#2b313a",
                    command=lambda n=name: self._select_profile(n),
                )
                btn.pack(fill="x", padx=4, pady=2)
                self.profile_buttons.append((btn, name))

        self.gallery_names, self.gallery_matrix = self.profile_mgr.get_gallery_matrix()

    def on_closing(self):
        self.is_streaming = False
        self.camera.stop()
        self.destroy()
