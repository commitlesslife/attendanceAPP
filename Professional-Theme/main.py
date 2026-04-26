# -*- coding: utf-8 -*-
"""
============================================================
  Smart Attendance System — Face Recognition
  College Lab Project | Python + OpenCV + CustomTkinter
============================================================
"""

import os
os.environ["PYTHONUTF8"] = "1"

import warnings
warnings.filterwarnings("ignore", message=".*CTkLabel Warning.*")
warnings.filterwarnings("ignore", message=".*pkg_resources.*")

import customtkinter as ctk
import cv2
import face_recognition
import numpy as np
from PIL import Image, ImageTk
import csv
import threading
from datetime import datetime
import tkinter.filedialog as filedialog
import json

# ──────────────────────────────────────────────────────────
#  CONFIGURATION
# ──────────────────────────────────────────────────────────

KNOWN_FACES_DIR = "known_faces"
STUDENTS_FILE = "students.json"
ATTENDANCE_FILE = "daily_attendance.csv"
CAMERA_INDEX = 0
FACE_MATCH_TOLERANCE = 0.50
FRAME_RESIZE_FACTOR = 0.25          # downscale for faster detection
PROCESS_EVERY_N_FRAMES = 3          # skip frames to save CPU

# Theme colours (Professional Obsidian & Sapphire)
CLR_BG_DARK   = "#0D0E15"
CLR_BG_CARD   = "#151722"
CLR_BG_CARD2  = "#1E2233"
CLR_ACCENT    = "#3B82F6"
CLR_ACCENT_HV = "#2563EB"
CLR_SUCCESS   = "#10B981"
CLR_DANGER    = "#EF4444"
CLR_TEXT       = "#F8FAFC"
CLR_TEXT_DIM   = "#94A3B8"
CLR_BORDER     = "#1E293B"


class AttendanceSystem(ctk.CTk):
    """Main application — drives camera, face recognition, and GUI."""

    def __init__(self):
        super().__init__()

        # ── window setup ──
        self.title("Smart Attendance System")
        self.geometry("1280x760")
        self.minsize(1100, 650)
        self.configure(fg_color=CLR_BG_DARK)
        ctk.set_appearance_mode("dark")

        # ── face-recognition state ──
        self.known_encodings: list[np.ndarray] = []
        self.known_names: list[str] = []
        self.known_roll_nos: list[str] = []
        self.known_ids: list[str] = []            # internal IDs (filename stems)
        self.current_faces: list[str] = []       # names visible right now
        self.current_roll_nos: list[str] = []    # roll nos visible right now
        self.current_confidences: list[float] = []  # match confidence %
        self.current_locations: list[tuple] = []  # bounding boxes

        # ── camera state ──
        self.cap: cv2.VideoCapture | None = None
        self.is_running = False
        self.is_reloading = False
        self.is_admin_open = False
        self.frame_count = 0
        self.latest_frame: np.ndarray | None = None

        # ── attendance state ──
        self.marked_today: set[str] = set()  # roll numbers marked today

        # ── load icons ──
        self.icons = {}
        icon_path = os.path.join(os.path.dirname(__file__), "assets", "icons")
        if os.path.exists(icon_path):
            for file in os.listdir(icon_path):
                if file.endswith('.png'):
                    name = file.split('.')[0]
                    img = Image.open(os.path.join(icon_path, file))
                    self.icons[name] = ctk.CTkImage(light_image=img, dark_image=img, size=(18, 18))

            logo_file = os.path.join(icon_path, "..", "logo.png")
            if os.path.exists(logo_file):
                logo_img = Image.open(logo_file)
                self.icons['logo'] = ctk.CTkImage(light_image=logo_img, dark_image=logo_img, size=(32, 32))
                try:
                    self.iconphoto(False, ImageTk.PhotoImage(logo_img))
                except Exception:
                    pass

        # ── build everything ──
        self._ensure_dirs()
        self._load_known_faces()
        self._load_todays_records()
        self._build_gui()
        self._start_camera()

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ──────────────────────────────────────────────────────
    #  FILE / FOLDER HELPERS
    # ──────────────────────────────────────────────────────

    @staticmethod
    def _ensure_dirs():
        """Create required directories if missing."""
        os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

    def _load_student_data(self) -> dict:
        """Load student details from students.json."""
        if os.path.isfile(STUDENTS_FILE):
            with open(STUDENTS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_student_data(self, data: dict):
        """Save student details to students.json."""
        with open(STUDENTS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _next_student_id(self, data: dict) -> str:
        """Get the next available numeric ID."""
        existing = [int(k) for k in data.keys() if k.isdigit()]
        return str(max(existing, default=0) + 1)

    def _load_known_faces(self):
        """Scan known_faces/ and encode every image, linking to students.json."""
        self.known_encodings.clear()
        self.known_names.clear()
        self.known_roll_nos.clear()
        self.known_ids.clear()
        valid_ext = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        student_data = self._load_student_data()

        if not os.path.isdir(KNOWN_FACES_DIR):
            return

        for fname in sorted(os.listdir(KNOWN_FACES_DIR)):
            if not fname.lower().endswith(valid_ext):
                continue
            
            sid = os.path.splitext(fname)[0]
            if sid not in student_data:
                continue

            path = os.path.join(KNOWN_FACES_DIR, fname)
            try:
                img = face_recognition.load_image_file(path)
                encs = face_recognition.face_encodings(img)
                if encs:
                    self.known_encodings.append(encs[0])
                    info = student_data[sid]
                    self.known_names.append(info.get("name", "Unknown"))
                    self.known_roll_nos.append(info.get("roll_no", "---"))
                    self.known_ids.append(sid)
            except Exception as exc:
                print(f"[WARN] Could not process {fname}: {exc}")

    def _load_todays_records(self):
        """Read daily_attendance.csv and populate self.marked_today to prevent duplicates."""
        self.marked_today.clear()
        if not os.path.isfile(ATTENDANCE_FILE):
            return
        
        today = datetime.now().strftime("%Y-%m-%d")
        with open(ATTENDANCE_FILE, "r", newline="", encoding="utf-8") as fh:
            reader = csv.reader(fh)
            next(reader, None)  # skip header
            for row in reader:
                # Format: Roll No, Name, Date, Time
                if len(row) >= 4 and row[2] == today:
                    self.marked_today.add(row[0])

    def _remove_student(self, sid: str):
        """Remove a student from students.json and delete their image."""
        data = self._load_student_data()
        if sid in data:
            name = data[sid].get("name", "Unknown")
            del data[sid]
            self._save_student_data(data)
            
            # Find and delete image
            valid_ext = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
            for ext in valid_ext:
                p = os.path.join(KNOWN_FACES_DIR, sid + ext)
                if os.path.isfile(p):
                    try:
                        os.remove(p)
                    except OSError:
                        pass
            
            self._on_reload_faces()
            self._show_toast(f"Removed {name}", CLR_DANGER)

    # ──────────────────────────────────────────────────────
    #  GUI LAYOUT
    # ──────────────────────────────────────────────────────

    def _build_gui(self):
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        self._build_sidebar()
        self._build_camera_panel()
        self._build_log_panel()

    # ── LEFT — SIDEBAR ──

    def _build_sidebar(self):
        sidebar = ctk.CTkFrame(self, width=280, corner_radius=0,
                               fg_color=CLR_BG_CARD, border_width=0)
        sidebar.grid(row=0, column=0, sticky="nswe")
        sidebar.grid_propagate(False)

        # — header —
        hdr = ctk.CTkFrame(sidebar, fg_color="transparent")
        hdr.pack(fill="x", padx=16, pady=(24, 16))
        
        ctk.CTkLabel(hdr, text=" Smart Attendance", image=self.icons.get('logo'), compound="left",
                     font=ctk.CTkFont(size=18, weight="bold"),
                     text_color=CLR_TEXT).pack(anchor="w")
        ctk.CTkLabel(hdr, text="Face Recognition System",
                     font=ctk.CTkFont(size=12),
                     text_color=CLR_TEXT_DIM).pack(anchor="w", padx=(38, 0))

        ctk.CTkFrame(sidebar, height=1, fg_color=CLR_BORDER).pack(fill="x", padx=16, pady=4)

        # — stats —
        stats = ctk.CTkFrame(sidebar, fg_color="transparent")
        stats.pack(fill="x", padx=16, pady=16)

        self.lbl_enrolled = ctk.CTkLabel(stats, text=f"Total enrolled: {len(self.known_names)}",
                                         font=ctk.CTkFont(size=13),
                                         text_color=CLR_TEXT_DIM)
        self.lbl_enrolled.pack(anchor="w", padx=14, pady=(0, 6))

        self.lbl_present = ctk.CTkLabel(stats, text=f"Present today: {len(self.marked_today)}",
                                        font=ctk.CTkFont(size=13),
                                        text_color=CLR_SUCCESS)
        self.lbl_present.pack(anchor="w", padx=14, pady=(0, 10))

        # — reload button —
        ctk.CTkButton(sidebar, text="Reload Faces", image=self.icons.get('refresh'),
                      font=ctk.CTkFont(size=13),
                      fg_color=CLR_BG_CARD2, hover_color=CLR_BORDER,
                      border_width=1, border_color=CLR_BORDER,
                      corner_radius=8,
                      command=self._on_reload_faces).pack(fill="x", padx=16, pady=(0, 6))

        # — admin panel (login required) —
        ctk.CTkButton(sidebar, text="Manage Students", image=self.icons.get('manage'),
                      font=ctk.CTkFont(size=13, weight="bold"),
                      fg_color=CLR_ACCENT, hover_color=CLR_ACCENT_HV,
                      corner_radius=8,
                      command=self._open_admin_login).pack(fill="x", padx=16, pady=(0, 20))

        # — student list section —
        list_lbl = ctk.CTkLabel(sidebar, text="ENROLLED STUDENTS",
                                font=ctk.CTkFont(size=11, weight="bold"),
                                text_color=CLR_TEXT_DIM)
        list_lbl.pack(anchor="w", padx=24, pady=(10, 4))

        self.student_list_frame = ctk.CTkScrollableFrame(
            sidebar, fg_color=CLR_BG_DARK, corner_radius=8,
            scrollbar_button_color=CLR_ACCENT)
        self.student_list_frame.pack(fill="both", expand=True, padx=16, pady=(0, 16))

        self._refresh_student_list()

    def _refresh_student_list(self):
        """Repopulate the sidebar student list."""
        for w in self.student_list_frame.winfo_children():
            w.destroy()

        if not self.known_names:
            ctk.CTkLabel(self.student_list_frame, text="No faces enrolled.",
                         text_color=CLR_TEXT_DIM).pack(pady=20)
            return

        for name, roll in zip(self.known_names, self.known_roll_nos):
            row = ctk.CTkFrame(self.student_list_frame, fg_color="transparent")
            row.pack(fill="x", pady=2)
            # coloured dot
            dot_clr = CLR_SUCCESS if roll in self.marked_today else CLR_TEXT_DIM
            ctk.CTkLabel(row, text="●", text_color=dot_clr,
                         font=ctk.CTkFont(size=10)).pack(side="left", padx=(6, 4))
            ctk.CTkLabel(row, text=roll, font=ctk.CTkFont(size=11),
                         text_color=CLR_ACCENT, width=36).pack(side="left", padx=(0, 4))
            ctk.CTkLabel(row, text=name, font=ctk.CTkFont(size=13),
                         text_color=CLR_TEXT).pack(side="left")

    def _update_stats(self):
        self.lbl_enrolled.configure(text=f"Total enrolled: {len(self.known_names)}")
        self.lbl_present.configure(text=f"Present today: {len(self.marked_today)}")

    # ── CENTRE — CAMERA PANEL ──

    def _build_camera_panel(self):
        centre = ctk.CTkFrame(self, fg_color=CLR_BG_DARK, corner_radius=0)
        centre.grid(row=0, column=1, sticky="nswe", padx=6, pady=6)
        centre.grid_rowconfigure(1, weight=1)
        centre.grid_columnconfigure(0, weight=1)

        # ── top bar ──
        topbar = ctk.CTkFrame(centre, fg_color="transparent")
        topbar.grid(row=0, column=0, sticky="we", padx=8, pady=(12, 4))

        ctk.CTkLabel(topbar, text="LIVE CAMERA FEED",
                     font=ctk.CTkFont(size=12, weight="bold"),
                     text_color=CLR_TEXT_DIM).pack(side="left")

        self.lbl_cam_status = ctk.CTkLabel(topbar, text="● Camera Active",
                                           font=ctk.CTkFont(size=12),
                                           text_color=CLR_SUCCESS)
        self.lbl_cam_status.pack(side="right")

        # ── camera label ──
        cam_border = ctk.CTkFrame(centre, fg_color=CLR_BG_CARD, corner_radius=14,
                                  border_width=1, border_color=CLR_BORDER)
        cam_border.grid(row=1, column=0, sticky="nswe", padx=8, pady=4)
        cam_border.grid_rowconfigure(0, weight=1)
        cam_border.grid_columnconfigure(0, weight=1)

        self.camera_label = ctk.CTkLabel(cam_border, text="",
                                         fg_color=CLR_BG_DARK, corner_radius=10)
        self.camera_label.grid(row=0, column=0, sticky="nswe", padx=6, pady=6)

        # ── bottom controls ──
        controls = ctk.CTkFrame(centre, fg_color="transparent")
        controls.grid(row=2, column=0, sticky="we", padx=8, pady=(8, 14))

        self.btn_mark = ctk.CTkButton(
            controls, text="Mark Attendance", image=self.icons.get('check'),
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color=CLR_SUCCESS, hover_color="#059669",
            height=44, corner_radius=8,
            command=self._on_mark_attendance,
        )
        self.btn_mark.pack(side="left", expand=True, fill="x", padx=(0, 6))

        self.btn_camera = ctk.CTkButton(
            controls, text="Pause Camera", image=self.icons.get('pause'),
            font=ctk.CTkFont(size=14),
            width=100, height=44,
            fg_color=CLR_BG_CARD2, hover_color=CLR_BORDER,
            border_width=1, border_color=CLR_BORDER,
            corner_radius=8,
            command=self._toggle_camera,
        )
        self.btn_camera.pack(side="right")

        # ── status toast ──
        self.toast_label = ctk.CTkLabel(
            centre, text="", font=ctk.CTkFont(size=13, weight="bold"),
            fg_color=CLR_BG_CARD2, corner_radius=8,
            text_color=CLR_SUCCESS, height=0,
        )

    # ── RIGHT — ATTENDANCE LOG ──

    def _build_log_panel(self):
        panel = ctk.CTkFrame(self, width=310, corner_radius=0,
                             fg_color=CLR_BG_CARD, border_width=0)
        panel.grid(row=0, column=2, sticky="nswe")
        panel.grid_propagate(False)

        # — header —
        hdr = ctk.CTkFrame(panel, fg_color="transparent")
        hdr.pack(fill="x", padx=18, pady=(24, 4))

        ctk.CTkLabel(hdr, text="TODAY'S ATTENDANCE",
                     font=ctk.CTkFont(size=12, weight="bold"),
                     text_color=CLR_TEXT_DIM).pack(anchor="w")
        ctk.CTkLabel(hdr, text=datetime.now().strftime("%A, %d %B %Y"),
                     font=ctk.CTkFont(size=13),
                     text_color=CLR_TEXT).pack(anchor="w", pady=(2, 0))

        ctk.CTkFrame(panel, height=1, fg_color=CLR_BORDER).pack(fill="x", padx=18, pady=12)

        # — scrollable log —
        self.log_frame = ctk.CTkScrollableFrame(
            panel, fg_color=CLR_BG_DARK, corner_radius=8,
            scrollbar_button_color=CLR_ACCENT)
        self.log_frame.pack(fill="both", expand=True, padx=14, pady=(0, 10))

        self._refresh_log()

        # — export button —
        ctk.CTkButton(panel, text="Export CSV", image=self.icons.get('export'),
                      font=ctk.CTkFont(size=12),
                      fg_color=CLR_BG_CARD2, hover_color=CLR_BORDER,
                      border_width=1, border_color=CLR_BORDER,
                      corner_radius=8,
                      command=self._on_export).pack(fill="x", padx=14, pady=(0, 18))

    def _refresh_log(self):
        """Repopulate the attendance log from today's records."""
        for w in self.log_frame.winfo_children():
            w.destroy()

        today = datetime.now().strftime("%Y-%m-%d")
        records: list[tuple[str, str, str]] = []  # (roll, name, time)

        if os.path.isfile(ATTENDANCE_FILE):
            with open(ATTENDANCE_FILE, "r", newline="", encoding="utf-8") as fh:
                reader = csv.reader(fh)
                next(reader, None)
                for row in reader:
                    if len(row) >= 4 and row[2] == today:
                        records.append((row[0], row[1], row[3]))

        if not records:
            ctk.CTkLabel(self.log_frame,
                         text="No attendance recorded yet.",
                         font=ctk.CTkFont(size=12),
                         text_color=CLR_TEXT_DIM).pack(pady=20)
            return

        for roll, name, time_str in records:
            card = ctk.CTkFrame(self.log_frame, fg_color=CLR_BG_CARD2, corner_radius=8)
            card.pack(fill="x", pady=3, padx=4)

            info_frame = ctk.CTkFrame(card, fg_color="transparent")
            info_frame.pack(side="left", padx=(10, 0), pady=6)
            ctk.CTkLabel(info_frame, text=f"{name}",
                         font=ctk.CTkFont(size=13, weight="bold"),
                         text_color=CLR_SUCCESS).pack(anchor="w")
            ctk.CTkLabel(info_frame, text=f"Roll: {roll}",
                         font=ctk.CTkFont(size=11),
                         text_color=CLR_TEXT_DIM).pack(anchor="w")
            ctk.CTkLabel(card, text=time_str,
                         font=ctk.CTkFont(size=12),
                         text_color=CLR_TEXT_DIM).pack(side="right", padx=12, pady=8)

    # ──────────────────────────────────────────────────────
    #  CAMERA  &  FACE RECOGNITION
    # ──────────────────────────────────────────────────────

    def _start_camera(self):
        """Open the default camera and begin the update loop."""
        try:
            self.cap = cv2.VideoCapture(CAMERA_INDEX)
            if not self.cap.isOpened():
                raise RuntimeError("Cannot open camera")
            self.is_running = True
            self.lbl_cam_status.configure(text="● Camera Active", text_color=CLR_SUCCESS)
            self._update_frame()
        except Exception as exc:
            self.is_running = False
            self.lbl_cam_status.configure(text="● Camera Error", text_color=CLR_DANGER)
            self.camera_label.configure(
                text=f"Camera not accessible.\n{exc}",
                font=ctk.CTkFont(size=14), text_color=CLR_DANGER)

    def _update_frame(self):
        """Grab a frame, run face detection periodically, and render."""
        if not self.is_running:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.after(30, self._update_frame)
            return

        self.frame_count += 1
        self.latest_frame = frame.copy()

        if getattr(self, "is_admin_open", False):
            self.camera_label.configure(image="", text="Camera paused for Admin Panel", 
                                        font=ctk.CTkFont(size=18), text_color=CLR_TEXT_DIM)
            self.after(30, self._update_frame)
            return

        self.camera_label.configure(text="")

        if not self.is_reloading and self.frame_count % PROCESS_EVERY_N_FRAMES == 0:
            small = cv2.resize(frame, (0, 0), fx=FRAME_RESIZE_FACTOR, fy=FRAME_RESIZE_FACTOR)
            rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

            locations = face_recognition.face_locations(rgb_small)
            encodings = face_recognition.face_encodings(rgb_small, locations)

            names = []
            roll_nos = []
            confidences = []
            scaled_locs = []

            for (top, right, bottom, left), enc in zip(locations, encodings):
                scale = int(1 / FRAME_RESIZE_FACTOR)
                top    *= scale
                right  *= scale
                bottom *= scale
                left   *= scale

                name = "Unknown"
                roll_no = ""
                confidence = 0.0
                if self.known_encodings:
                    matches = face_recognition.compare_faces(self.known_encodings, enc, tolerance=FACE_MATCH_TOLERANCE)
                    distances = face_recognition.face_distance(self.known_encodings, enc)

                    if True in matches:
                        best = np.argmin(distances)
                        if matches[best]:
                            name = self.known_names[best]
                            roll_no = self.known_roll_nos[best]
                            confidence = round((1 - distances[best]) * 100, 1)

                names.append(name)
                roll_nos.append(roll_no)
                confidences.append(confidence)
                scaled_locs.append((top, right, bottom, left))

            self.current_faces = names
            self.current_roll_nos = roll_nos
            self.current_confidences = confidences
            self.current_locations = scaled_locs

        display = frame.copy()
        for i, (name, (top, right, bottom, left)) in enumerate(zip(self.current_faces, self.current_locations)):
            is_known = name != "Unknown"
            colour = (16, 185, 129) if is_known else (239, 68, 68)

            cv2.rectangle(display, (left, top), (right, bottom), colour, 2)
            conf = self.current_confidences[i] if i < len(self.current_confidences) else 0
            label = f"{name} ({conf}%)" if is_known else "Unknown"
            label_h = 32
            cv2.rectangle(display, (left, bottom), (right, bottom + label_h), colour, cv2.FILLED)
            cv2.putText(display, label, (left + 6, bottom + 23),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)

        lbl_w = self.camera_label.winfo_width()
        lbl_h = self.camera_label.winfo_height()
        if lbl_w > 10 and lbl_h > 10:
            img.thumbnail((lbl_w, lbl_h), Image.LANCZOS)

        imgtk = ImageTk.PhotoImage(image=img)
        self.camera_label.configure(image=imgtk, text="")
        self.camera_label.imgtk = imgtk

        self.after(30, self._update_frame)

    def _toggle_camera(self):
        if self.is_running:
            self.is_running = False
            self.btn_camera.configure(text="Resume Camera", image=self.icons.get('camera'))
            self.lbl_cam_status.configure(text="○ Camera Paused", text_color=CLR_TEXT_DIM)
            self.camera_label.configure(image="", text="Camera paused", font=ctk.CTkFont(size=16), text_color=CLR_TEXT_DIM)
        else:
            self.is_running = True
            self.btn_camera.configure(text="Pause Camera", image=self.icons.get('pause'))
            self.lbl_cam_status.configure(text="● Camera Active", text_color=CLR_SUCCESS)
            self._update_frame()

    # ──────────────────────────────────────────────────────
    #  ATTENDANCE LOGIC
    # ──────────────────────────────────────────────────────

    def _log_attendance(self, name: str, roll_no: str):
        today = datetime.now().strftime("%Y-%m-%d")
        time_now = datetime.now().strftime("%H:%M:%S")
        file_exists = os.path.isfile(ATTENDANCE_FILE)
        with open(ATTENDANCE_FILE, "a", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            if not file_exists:
                writer.writerow(["Roll No", "Name", "Date", "Time"])
            writer.writerow([roll_no, name, today, time_now])
        self.marked_today.add(roll_no)

    def _on_mark_attendance(self):
        if not self.current_faces:
            self._show_toast("No faces detected in frame", CLR_DANGER)
            return

        recognised = [(n, r) for n, r in zip(self.current_faces, self.current_roll_nos) if n != "Unknown"]

        if not recognised:
            self._show_toast("No recognised students in frame", CLR_DANGER)
            return

        newly_marked = []
        already_marked = []

        for name, roll in recognised:
            if roll in self.marked_today:
                already_marked.append(name)
            else:
                self._log_attendance(name, roll)
                newly_marked.append(name)

        if newly_marked:
            self._show_toast(f"Marked: {', '.join(newly_marked)}", CLR_SUCCESS)
        elif already_marked:
            self._show_toast("Already marked today", CLR_TEXT_DIM)

        self._refresh_log()
        self._refresh_student_list()
        self._update_stats()

    def _on_export(self):
        if os.path.isfile(ATTENDANCE_FILE):
            full_path = os.path.abspath(ATTENDANCE_FILE)
            self._show_toast(f"CSV saved at:\n{full_path}", CLR_SUCCESS)
        else:
            self._show_toast("No attendance data yet", CLR_DANGER)

    # ──────────────────────────────────────────────────────
    #  ADMIN PANEL & LOGIN
    # ──────────────────────────────────────────────────────

    def _open_admin_login(self):
        login = ctk.CTkToplevel(self)
        login.title("Admin Login")
        login.geometry("320x220")
        login.configure(fg_color=CLR_BG_DARK)
        login.transient(self)
        login.grab_set()
        login.resizable(False, False)

        login.after(10, lambda: login.geometry(f"+{self.winfo_x() + self.winfo_width()//2 - 160}+{self.winfo_y() + self.winfo_height()//2 - 110}"))

        self.is_admin_open = True
        login.protocol("WM_DELETE_WINDOW", lambda: [setattr(self, 'is_admin_open', False), login.destroy()])

        ctk.CTkLabel(login, text="Admin Authentication", font=ctk.CTkFont(size=16, weight="bold"), text_color=CLR_TEXT).pack(pady=(20, 10))

        pwd_entry = ctk.CTkEntry(login, placeholder_text="Password", show="*", font=ctk.CTkFont(size=14), fg_color=CLR_BG_CARD2, border_color=CLR_BORDER, width=260, height=36)
        pwd_entry.pack(pady=(0, 10))

        status_lbl = ctk.CTkLabel(login, text="", font=ctk.CTkFont(size=12), text_color=CLR_DANGER)
        status_lbl.pack()

        def _verify(event=None):
            if pwd_entry.get() == "123":
                login.protocol("WM_DELETE_WINDOW", lambda: None)
                login.destroy()
                self._open_admin_panel()
            else:
                status_lbl.configure(text="Incorrect password")
                pwd_entry.delete(0, 'end')

        pwd_entry.bind("<Return>", _verify)
        ctk.CTkButton(login, text="Login", font=ctk.CTkFont(size=13, weight="bold"), fg_color=CLR_ACCENT, hover_color=CLR_ACCENT_HV, corner_radius=8, width=260, height=36, command=_verify).pack(pady=(0, 20))

    def _open_admin_panel(self):
        panel = ctk.CTkToplevel(self)
        panel.title("Admin - Manage Students")
        panel.geometry("600x520")
        panel.configure(fg_color=CLR_BG_DARK)
        panel.transient(self)
        panel.grab_set()
        panel.resizable(True, True)
        panel.minsize(500, 400)

        panel.after(10, lambda: panel.geometry(f"+{self.winfo_x() + self.winfo_width()//2 - 300}+{self.winfo_y() + self.winfo_height()//2 - 260}"))

        self.is_admin_open = True
        panel.protocol("WM_DELETE_WINDOW", lambda: [setattr(self, 'is_admin_open', False), panel.destroy()])

        header = ctk.CTkFrame(panel, fg_color="transparent")
        header.pack(fill="x", padx=24, pady=(20, 0))

        ctk.CTkLabel(header, text="Student Management", font=ctk.CTkFont(size=20, weight="bold"), text_color=CLR_TEXT).pack(side="left")

        ctk.CTkButton(header, text="Add Student", image=self.icons.get('add_user'), font=ctk.CTkFont(size=13, weight="bold"), fg_color=CLR_ACCENT, hover_color=CLR_ACCENT_HV, corner_radius=8, width=140, height=34, command=lambda: self._on_register_student(parent=panel, on_complete=lambda: _populate())).pack(side="right")

        col_hdr = ctk.CTkFrame(panel, fg_color=CLR_BG_CARD2, corner_radius=8)
        col_hdr.pack(fill="x", padx=24, pady=(16, 0))

        ctk.CTkLabel(col_hdr, text="ID", width=40, font=ctk.CTkFont(size=12, weight="bold"), text_color=CLR_TEXT_DIM).pack(side="left", padx=(14, 0), pady=8)
        ctk.CTkLabel(col_hdr, text="Roll No", width=80, font=ctk.CTkFont(size=12, weight="bold"), text_color=CLR_TEXT_DIM).pack(side="left", padx=(10, 0), pady=8)
        ctk.CTkLabel(col_hdr, text="Student Name", font=ctk.CTkFont(size=12, weight="bold"), text_color=CLR_TEXT_DIM).pack(side="left", padx=(10, 0), pady=8)
        ctk.CTkLabel(col_hdr, text="Action", width=80, font=ctk.CTkFont(size=12, weight="bold"), text_color=CLR_TEXT_DIM).pack(side="right", padx=(0, 14), pady=8)

        list_frame = ctk.CTkScrollableFrame(panel, fg_color=CLR_BG_CARD, corner_radius=8, scrollbar_button_color=CLR_ACCENT)
        list_frame.pack(fill="both", expand=True, padx=24, pady=(4, 12))

        footer = ctk.CTkFrame(panel, fg_color="transparent", height=30)
        footer.pack(fill="x", padx=24, pady=(0, 16))
        count_lbl = ctk.CTkLabel(footer, text="", font=ctk.CTkFont(size=12), text_color=CLR_TEXT_DIM)
        count_lbl.pack(side="left")

        def _populate():
            for w in list_frame.winfo_children():
                w.destroy()
            data = self._load_student_data()
            count_lbl.configure(text=f"Total Records: {len(data)}")
            if not data:
                ctk.CTkLabel(list_frame, text="No students found.", text_color=CLR_TEXT_DIM).pack(pady=20)
                return

            for sid, info in sorted(data.items(), key=lambda x: int(x[0]) if x[0].isdigit() else 0):
                row = ctk.CTkFrame(list_frame, fg_color="transparent")
                row.pack(fill="x", pady=4)
                
                ctk.CTkLabel(row, text=sid, width=40, text_color=CLR_TEXT_DIM, font=ctk.CTkFont(size=11)).pack(side="left", padx=(14, 0))
                ctk.CTkLabel(row, text=info.get("roll_no", ""), width=80, text_color=CLR_ACCENT, font=ctk.CTkFont(size=12, weight="bold")).pack(side="left", padx=(10, 0))
                ctk.CTkLabel(row, text=info.get("name", ""), text_color=CLR_TEXT, font=ctk.CTkFont(size=13)).pack(side="left", padx=(10, 0))
                
                def make_remove_cmd(sid_to_remove=sid):
                    return lambda: [_remove_student(sid_to_remove), _populate()]

                ctk.CTkButton(row, text="", image=self.icons.get('delete'), width=40, fg_color="transparent", hover_color=CLR_DANGER, command=make_remove_cmd()).pack(side="right", padx=(0, 14))

        _populate()

    def _on_register_student(self, parent=None, on_complete=None):
        dialog = ctk.CTkToplevel(parent if parent else self)
        dialog.title("Register New Student")
        dialog.geometry("460x640")
        dialog.configure(fg_color=CLR_BG_DARK)
        dialog.transient(parent if parent else self)
        dialog.grab_set()
        dialog.resizable(False, False)

        p_x = parent.winfo_x() if parent else self.winfo_x()
        p_y = parent.winfo_y() if parent else self.winfo_y()
        p_w = parent.winfo_width() if parent else self.winfo_width()
        p_h = parent.winfo_height() if parent else self.winfo_height()

        dialog.after(10, lambda: dialog.geometry(f"+{p_x + p_w//2 - 230}+{p_y + p_h//2 - 320}"))

        ctk.CTkLabel(dialog, text="Register New Student", font=ctk.CTkFont(size=18, weight="bold"), text_color=CLR_TEXT).pack(pady=(20, 10))

        form = ctk.CTkFrame(dialog, fg_color="transparent")
        form.pack(fill="x", padx=30)

        ctk.CTkLabel(form, text="Roll Number:", font=ctk.CTkFont(size=13), text_color=CLR_TEXT_DIM).pack(anchor="w")
        roll_entry = ctk.CTkEntry(form, placeholder_text="e.g. CS101", font=ctk.CTkFont(size=14), fg_color=CLR_BG_CARD2, border_color=CLR_BORDER, height=36)
        roll_entry.pack(fill="x", pady=(2, 8))

        ctk.CTkLabel(form, text="Student Name:", font=ctk.CTkFont(size=13), text_color=CLR_TEXT_DIM).pack(anchor="w")
        name_entry = ctk.CTkEntry(form, placeholder_text="e.g. Sunil Dehru", font=ctk.CTkFont(size=14), fg_color=CLR_BG_CARD2, border_color=CLR_BORDER, height=36)
        name_entry.pack(fill="x", pady=(2, 8))

        cam_frame = ctk.CTkFrame(dialog, width=320, height=240, fg_color="black", corner_radius=8)
        cam_frame.pack(pady=10)
        cam_frame.pack_propagate(False)
        cam_lbl = ctk.CTkLabel(cam_frame, text="Camera Preview\nWill appear here", text_color=CLR_TEXT_DIM)
        cam_lbl.pack(expand=True, fill="both")

        status_lbl = ctk.CTkLabel(dialog, text="", font=ctk.CTkFont(size=13), text_color=CLR_DANGER)
        status_lbl.pack()

        btn_frame = ctk.CTkFrame(dialog, fg_color="transparent")
        btn_frame.pack(fill="x", padx=30, pady=(4, 10))

        def _save_student(image_path_or_frame, from_webcam=False):
            name = name_entry.get().strip()
            roll = roll_entry.get().strip()
            data = self._load_student_data()
            sid = self._next_student_id(data)
            data[sid] = {"name": name, "roll_no": roll}
            self._save_student_data(data)
            
            if from_webcam:
                cv2.imwrite(os.path.join(KNOWN_FACES_DIR, f"{sid}.jpg"), image_path_or_frame)
            else:
                import shutil
                ext = os.path.splitext(image_path_or_frame)[1]
                shutil.copy2(image_path_or_frame, os.path.join(KNOWN_FACES_DIR, f"{sid}{ext}"))
            
            dialog.destroy()
            self._on_reload_faces()
            self._show_toast(f"Registered: {roll} - {name}", CLR_SUCCESS)
            if on_complete:
                on_complete()

        def _capture_loop(good_frames):
            if not dialog.winfo_exists():
                return
            frame = self.latest_frame
            if frame is None:
                dialog.after(30, lambda: _capture_loop(good_frames))
                return
            
            small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            locs = face_recognition.face_locations(rgb)
            display = frame.copy()
            
            if len(locs) == 1:
                top, right, bottom, left = [v * 2 for v in locs[0]]
                cv2.rectangle(display, (left, top), (right, bottom), (16, 185, 129), 2)
                good_frames += 1
                status_lbl.configure(text=f"Hold still... {good_frames}/5", text_color=CLR_SUCCESS)
            else:
                good_frames = 0
                if len(locs) > 1:
                    status_lbl.configure(text="Multiple faces detected!", text_color=CLR_DANGER)
                else:
                    status_lbl.configure(text="No face detected...", text_color=CLR_TEXT_DIM)
            
            preview = cv2.resize(display, (320, 240))
            preview_rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(preview_rgb)
            ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(320, 240))
            cam_lbl.configure(image=ctk_img, text="")
            cam_lbl.image = ctk_img
            
            if good_frames >= 5:
                status_lbl.configure(text="Captured!", text_color=CLR_SUCCESS)
                _save_student(frame, from_webcam=True)
            else:
                dialog.after(50, lambda: _capture_loop(good_frames))

        def _start_capture():
            if not name_entry.get().strip() or not roll_entry.get().strip():
                status_lbl.configure(text="Please fill both fields first")
                return
            btn_frame.pack_forget()
            roll_entry.configure(state="disabled")
            name_entry.configure(state="disabled")
            status_lbl.configure(text="Looking for a face...", text_color=CLR_TEXT)
            _capture_loop(0)

        def _browse():
            if not name_entry.get().strip() or not roll_entry.get().strip():
                status_lbl.configure(text="Please fill both fields first")
                return
            path = filedialog.askopenfilename(title="Select face photo", filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp")])
            if not path:
                return
            _save_student(path, from_webcam=False)

        ctk.CTkButton(btn_frame, text="Auto-Capture", image=self.icons.get('camera'), font=ctk.CTkFont(size=13, weight="bold"), fg_color=CLR_ACCENT, hover_color=CLR_ACCENT_HV, corner_radius=8, height=38, command=_start_capture).pack(side="left", expand=True, fill="x", padx=(0, 4))
        ctk.CTkButton(btn_frame, text="Browse File", image=self.icons.get('folder'), font=ctk.CTkFont(size=13), fg_color=CLR_BG_CARD2, hover_color=CLR_BORDER, border_width=1, border_color=CLR_BORDER, corner_radius=8, height=38, command=_browse).pack(side="right", expand=True, fill="x", padx=(4, 0))

    def _on_reload_faces(self):
        """Re-scan the known_faces directory and reload encodings."""
        self._show_toast("Reloading faces…", CLR_TEXT_DIM)
        self.is_reloading = True

        def _reload():
            new_encodings = []
            new_names = []
            new_roll_nos = []
            new_ids = []
            valid_ext = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
            student_data = self._load_student_data()
            if os.path.isdir(KNOWN_FACES_DIR):
                for fname in sorted(os.listdir(KNOWN_FACES_DIR)):
                    if not fname.lower().endswith(valid_ext):
                        continue
                    sid = os.path.splitext(fname)[0]
                    if sid not in student_data:
                        continue
                    path = os.path.join(KNOWN_FACES_DIR, fname)
                    try:
                        img = face_recognition.load_image_file(path)
                        encs = face_recognition.face_encodings(img)
                        if encs:
                            new_encodings.append(encs[0])
                            info = student_data[sid]
                            new_names.append(info.get("name", "Unknown"))
                            new_roll_nos.append(info.get("roll_no", "---"))
                            new_ids.append(sid)
                    except Exception as exc:
                        print(f"[WARN] Could not process {fname}: {exc}")
            self.after(0, lambda: self._finish_reload(new_encodings, new_names, new_roll_nos, new_ids))

        threading.Thread(target=_reload, daemon=True).start()

    def _finish_reload(self, new_encodings, new_names, new_roll_nos, new_ids):
        self.known_encodings = new_encodings
        self.known_names = new_names
        self.known_roll_nos = new_roll_nos
        self.known_ids = new_ids
        self.is_reloading = False
        self._refresh_student_list()
        self._update_stats()
        self._show_toast(f"Loaded {len(self.known_names)} student(s)", CLR_SUCCESS)

    def _show_toast(self, message: str, color: str):
        self.toast_label.configure(text=message, text_color=color, height=40)
        self.toast_label.place(relx=0.5, rely=0.08, anchor="n")
        self.after(3000, lambda: self.toast_label.place_forget())

    def _on_close(self):
        self.is_running = False
        if self.cap:
            self.cap.release()
        self.destroy()


if __name__ == "__main__":
    try:
        app = AttendanceSystem()
        app.mainloop()
    except Exception as e:
        with open("error.log", "a") as f:
            f.write(f"{datetime.now()}: {e}\n")
        import traceback
        traceback.print_exc()
