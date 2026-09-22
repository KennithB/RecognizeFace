from datetime import datetime
from pathlib import Path
import time
from tkinter import messagebox
from typing import Dict, List, Optional
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

        # Window settings
        self.title("Face Recognition & Stranger Monitor")
        self.geometry("1280x800")
        self.minsize(1050, 680)

        # Default appearance
        self.current_theme = "dark"
        ctk.set_appearance_mode(self.current_theme)
        ctk.set_default_color_theme("blue")

        # Storage & engine
        self.strangers_dir = Path("captured_strangers")
        self.strangers_dir.mkdir(parents=True, exist_ok=True)

        self.profile_mgr = ProfileManager(profiles_dir="profiles")
        self.engine = FaceRecognitionEngine(model_name="buffalo_l", use_gpu=True)
        self.camera = CameraStream(source=0)

        # Runtime states
        self.is_streaming = False
        self.current_frame: Optional[np.ndarray] = None
        self.gallery_names, self.gallery_matrix = self.profile_mgr.get_gallery_matrix()
        self.matching_threshold = 0.45

        # Stranger tracking & de-duplication
        self.recent_strangers: List[Dict] = []
        self.stranger_similarity_thresh = 0.60

        self.prev_time = time.time()
        self.fps = 0.0

        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        self._build_layout()

    def _build_layout(self):
        # Main root box with padding
        main_box = ctk.CTkFrame(self, fg_color="transparent")
        main_box.pack(fill="both", expand=True, padx=14, pady=14)

        # ================= Left: Camera Stream =================
        self.stream_panel = ctk.CTkFrame(main_box, fg_color=("#e6e9ee", "#181a1f"), corner_radius=8)
        self.stream_panel.pack(side="left", fill="both", expand=True, padx=(0, 8))

        # Stream header
        header = ctk.CTkFrame(self.stream_panel, fg_color="transparent")
        header.pack(fill="x", padx=14, pady=12)

        self.lbl_stream_status = ctk.CTkLabel(
            header,
            text="Camera Standby",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#8a94a6",
        )
        self.lbl_stream_status.pack(side="left")

        # Right side of stream header: FPS and Theme Toggle Switch
        header_right = ctk.CTkFrame(header, fg_color="transparent")
        header_right.pack(side="right")

        self.lbl_fps = ctk.CTkLabel(
            header_right,
            text="0.0 FPS",
            font=ctk.CTkFont(size=13),
            text_color="#8a94a6",
        )
        self.lbl_fps.pack(side="left", padx=(0, 14))

        # Clean toggle switch (No sun/moon icons)
        self.theme_switch = ctk.CTkSwitch(
            header_right,
            text="Dark Mode",
            font=ctk.CTkFont(size=12),
            command=self._toggle_theme,
            onvalue=1,
            offvalue=0,
        )
        self.theme_switch.select()  # Start in dark mode
        self.theme_switch.pack(side="left")

        # Video Screen with smooth corners
        self.video_display = ctk.CTkLabel(
            self.stream_panel,
            text="Camera inactive\nClick 'Start Camera' to monitor feed",
            font=ctk.CTkFont(size=14),
            fg_color=("#d2d6de", "#0e1013"),
            corner_radius=6,
        )
        self.video_display.pack(fill="both", expand=True, padx=14, pady=(0, 14))

        # ================= Right: Controls & Stranger Detections =================
        self.right_panel = ctk.CTkFrame(main_box, width=380, fg_color=("#e6e9ee", "#181a1f"), corner_radius=8)
        self.right_panel.pack(side="right", fill="y", padx=(8, 0))
        self.right_panel.pack_propagate(False)

        # Camera connection row
        cam_bar = ctk.CTkFrame(self.right_panel, fg_color="transparent")
        cam_bar.pack(fill="x", padx=14, pady=12)

        self.entry_source = ctk.CTkEntry(
            cam_bar,
            placeholder_text="0 or RTSP stream URL",
            height=34,
            corner_radius=6,
        )
        self.entry_source.insert(0, "0")
        self.entry_source.pack(side="left", fill="x", expand=True, padx=(0, 8))

        self.btn_camera = ctk.CTkButton(
            cam_bar,
            text="Start Camera",
            width=100,
            height=34,
            corner_radius=6,
            command=self._toggle_camera,
        )
        self.btn_camera.pack(side="right")

        # Unrecognized Section Title & Clear button
        sec_header = ctk.CTkFrame(self.right_panel, fg_color="transparent")
        sec_header.pack(fill="x", padx=14, pady=(4, 6))

        self.lbl_unrec_title = ctk.CTkLabel(
            sec_header,
            text="Unrecognized Faces",
            font=ctk.CTkFont(size=13, weight="bold"),
            anchor="w",
        )
        self.lbl_unrec_title.pack(side="left")

        btn_clear = ctk.CTkButton(
            sec_header,
            text="Clear",
            width=60,
            height=26,
            corner_radius=6,
            font=ctk.CTkFont(size=11),
            fg_color=("#cbd2dc", "#2b313a"),
            text_color=("#111827", "#f9fafb"),
            hover_color=("#b6bfcb", "#363d48"),
            command=self._clear_strangers,
        )
        btn_clear.pack(side="right")

        # Scrollable feed for captured unrecognized faces
        self.stranger_scroll = ctk.CTkScrollableFrame(
            self.right_panel,
            fg_color=("#f3f5f8", "#121417"),
            corner_radius=6,
        )
        self.stranger_scroll.pack(fill="both", expand=True, padx=14, pady=(0, 14))

        self.lbl_no_strangers = ctk.CTkLabel(
            self.stranger_scroll,
            text="No unrecognized faces detected yet",
            font=ctk.CTkFont(size=12),
            text_color="#6c757d",
        )
        self.lbl_no_strangers.pack(pady=30)

    def _toggle_theme(self):
        if self.theme_switch.get() == 1:
            self.current_theme = "dark"
            self.theme_switch.configure(text="Dark Mode")
        else:
            self.current_theme = "light"
            self.theme_switch.configure(text="Light Mode")
        ctk.set_appearance_mode(self.current_theme)

    def _toggle_camera(self):
        if not self.is_streaming:
            source = self.entry_source.get().strip()
            self.camera.source = source
            if not self.camera.start():
                messagebox.showerror("Error", f"Failed to connect to camera source: '{source}'")
                return
            self.is_streaming = True
            self.btn_camera.configure(text="Stop Camera", fg_color="#992d22", hover_color="#b83528")
            self.lbl_stream_status.configure(text="Live Feed Active", text_color="#2ecc71")
            self._update_loop()
        else:
            self.is_streaming = False
            self.camera.stop()
            self.btn_camera.configure(text="Start Camera", fg_color=["#3B8ED0", "#1F6AA5"], hover_color=["#36719F", "#144870"])
            self.lbl_stream_status.configure(text="Camera Standby", text_color="#8a94a6")
            self.lbl_fps.configure(text="0.0 FPS")
            self.video_display.configure(image="", text="Camera inactive\nClick 'Start Camera' to monitor feed")

    def _crop_face_high_quality(self, frame: np.ndarray, bbox: np.ndarray) -> Optional[np.ndarray]:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox.astype(int)

        bw = x2 - x1
        bh = y2 - y1
        pad_x = int(bw * 0.35)
        pad_y = int(bh * 0.35)

        cx1 = max(0, x1 - pad_x)
        cy1 = max(0, y1 - pad_y)
        cx2 = min(w, x2 + pad_x)
        cy2 = min(h, y2 + pad_y)

        if cx2 > cx1 and cy2 > cy1:
            return frame[cy1:cy2, cx1:cx2].copy()
        return None

    def _handle_unrecognized_face(self, frame: np.ndarray, face) -> None:
        embedding = face.normed_embedding
        now = time.time()

        for item in self.recent_strangers:
            sim = float(np.dot(item["embedding"], embedding))
            if sim >= self.stranger_similarity_thresh:
                item["last_seen"] = now
                return

        crop = self._crop_face_high_quality(frame, face.bbox)
        if crop is None:
            return

        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        time_display = datetime.now().strftime("%I:%M:%S %p")
        image_filename = f"stranger_{timestamp_str}_{int(now % 1000)}.jpg"
        save_path = self.strangers_dir / image_filename

        cv2.imwrite(str(save_path), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        stranger_record = {
            "id": image_filename,
            "embedding": embedding,
            "last_seen": now,
            "image_path": save_path,
        }
        self.recent_strangers.insert(0, stranger_record)
        self._add_stranger_card(stranger_record, crop, time_display)

    def _add_stranger_card(self, record: Dict, crop_bgr: np.ndarray, time_str: str):
        if self.lbl_no_strangers.winfo_ismapped():
            self.lbl_no_strangers.pack_forget()

        card = ctk.CTkFrame(self.stranger_scroll, fg_color=("#ffffff", "#1d2026"), corner_radius=6)
        card.pack(fill="x", pady=5, padx=2)

        content_box = ctk.CTkFrame(card, fg_color="transparent")
        content_box.pack(fill="x", padx=10, pady=10)

        # Left: Face thumbnail with smooth edges
        thumb_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        thumb_pil = Image.fromarray(thumb_rgb).resize((76, 76), Image.Resampling.LANCZOS)
        thumb_tk = ctk.CTkImage(light_image=thumb_pil, dark_image=thumb_pil, size=(76, 76))

        lbl_img = ctk.CTkLabel(content_box, image=thumb_tk, text="", corner_radius=4)
        lbl_img.image = thumb_tk
        lbl_img.pack(side="left", padx=(0, 10))

        # Right: Details & Action controls
        details = ctk.CTkFrame(content_box, fg_color="transparent")
        details.pack(side="left", fill="both", expand=True)

        lbl_time = ctk.CTkLabel(
            details,
            text=f"Detected: {time_str}",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#e03131",
            anchor="w",
        )
        lbl_time.pack(fill="x", pady=(0, 4))

        entry_name = ctk.CTkEntry(
            details,
            placeholder_text="Enter name to train...",
            height=28,
            corner_radius=6,
            font=ctk.CTkFont(size=12),
        )
        entry_name.pack(fill="x", pady=(0, 6))

        actions = ctk.CTkFrame(details, fg_color="transparent")
        actions.pack(fill="x")

        btn_train = ctk.CTkButton(
            actions,
            text="Train Face",
            height=26,
            corner_radius=6,
            font=ctk.CTkFont(size=11, weight="bold"),
            fg_color="#2b78e4",
            hover_color="#1e5fbf",
            command=lambda r=record, e=entry_name, c=card: self._train_stranger(r, e, c),
        )
        btn_train.pack(side="left", fill="x", expand=True, padx=(0, 6))

        btn_dismiss = ctk.CTkButton(
            actions,
            text="Dismiss",
            width=55,
            height=26,
            corner_radius=6,
            font=ctk.CTkFont(size=11),
            fg_color=("#cbd2dc", "#363c46"),
            text_color=("#111827", "#f9fafb"),
            hover_color=("#b6bfcb", "#444b57"),
            command=lambda r=record, c=card: self._dismiss_stranger(r, c),
        )
        btn_dismiss.pack(side="right")

        record["card_frame"] = card

    def _train_stranger(self, record: Dict, entry_name: ctk.CTkEntry, card: ctk.CTkFrame):
        name = entry_name.get().strip()
        if not name:
            messagebox.showwarning("Input Needed", "Type a person name to train this face.")
            return

        self.profile_mgr.save_profile(name, record["embedding"])
        self.gallery_names, self.gallery_matrix = self.profile_mgr.get_gallery_matrix()

        self._dismiss_stranger(record, card)
        messagebox.showinfo("Trained", f"Face trained and registered as '{name}'.")

    def _dismiss_stranger(self, record: Dict, card: ctk.CTkFrame):
        card.destroy()
        if record in self.recent_strangers:
            self.recent_strangers.remove(record)

        if not self.stranger_scroll.winfo_children():
            self.lbl_no_strangers.pack(pady=30)

    def _clear_strangers(self):
        for widget in self.stranger_scroll.winfo_children():
            widget.destroy()
        self.recent_strangers.clear()
        self.lbl_no_strangers = ctk.CTkLabel(
            self.stranger_scroll,
            text="No unrecognized faces detected yet",
            font=ctk.CTkFont(size=12),
            text_color="#6c757d",
        )
        self.lbl_no_strangers.pack(pady=30)

    def _update_loop(self):
        if not self.is_streaming:
            return

        frame = self.camera.read_frame()
        if frame is not None:
            self.current_frame = frame.copy()
            faces = self.engine.analyze_frame(frame)
            stranger_in_frame = 0

            for face in faces:
                x1, y1, x2, y2 = face.bbox.astype(int)
                name, score = self.engine.match_embedding(
                    face.normed_embedding,
                    self.gallery_names,
                    self.gallery_matrix,
                    threshold=self.matching_threshold,
                )

                if name == "STRANGER":
                    stranger_in_frame += 1
                    box_color = (0, 0, 255)
                    label_text = f"UNRECOGNIZED ({score:.2f})"
                    self._handle_unrecognized_face(frame, face)
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

            # Update FPS & Stream Status
            now = time.time()
            dt = now - self.prev_time
            self.prev_time = now
            if dt > 0:
                self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt)
            self.lbl_fps.configure(text=f"{self.fps:.1f} FPS")

            if stranger_in_frame > 0:
                self.lbl_stream_status.configure(
                    text=f"Alert: {stranger_in_frame} Unrecognized Face(s)",
                    text_color="#ff5252",
                )
            else:
                self.lbl_stream_status.configure(
                    text=f"Monitoring ({len(faces)} Known Detected)" if faces else "Monitoring (No Faces)",
                    text_color="#2ecc71" if faces else "#8a94a6",
                )

            # Render video to viewport
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

    def on_closing(self):
        self.is_streaming = False
        self.camera.stop()
        self.destroy()
