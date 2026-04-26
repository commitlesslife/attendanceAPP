# 📸 Smart Attendance System — Face Recognition

A modern desktop application that automates classroom attendance using real-time facial recognition. Built with Python, OpenCV, and CustomTkinter.

---

## ✨ Features

| Feature | Description |
|---|---|
| **Real-time Face Detection** | Live webcam feed with bounding boxes drawn around detected faces |
| **Face Recognition** | Matches detected faces against a pre-enrolled student database |
| **Visual Feedback** | 🟢 Green box for recognised students, 🔴 Red box for unknown faces |
| **One-Click Attendance** | Single button press logs all recognised students in the frame |
| **Duplicate Prevention** | Same student cannot be marked twice on the same day |
| **Daily CSV Log** | All records are saved to `daily_attendance.csv` with Name, Date, and Time |
| **Auto-Reload** | Re-scan the `known_faces/` folder without restarting the app |
| **Pause / Resume** | Pause the camera feed at any time |
| **Dark-Themed GUI** | Modern, sleek interface powered by CustomTkinter |

---

## 📁 Project Structure

```
Python-Project/
├── main.py                 # Main application (entry point)
├── requirements.txt        # Python dependencies
├── daily_attendance.csv    # Auto-generated attendance log
├── known_faces/            # Student face images (filename = name)
│   ├── John_Doe.jpg
│   ├── Jane_Smith.png
│   └── ...
└── README.md               # This file
```

---

## 🛠️ Installation

### Prerequisites

- **Python 3.9+** (recommended: 3.10 or 3.11)
- A working **webcam**
- **CMake** (required to build dlib)
- **Visual Studio Build Tools** (Windows) or **gcc/g++** (Linux/Mac)

### Step 1 — Install CMake

**Windows:**
```bash
# Option A: Using winget
winget install Kitware.CMake

# Option B: Download from https://cmake.org/download/
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get update
sudo apt-get install cmake build-essential
```

**macOS:**
```bash
brew install cmake
```

### Step 2 — Install Python Dependencies

```bash
# (Recommended) Create a virtual environment first
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install all dependencies
pip install -r requirements.txt
```

> **⚠️ Troubleshooting `dlib` / `face-recognition` installation on Windows:**
>
> If `pip install face-recognition` fails, try these steps:
>
> 1. Install **Visual Studio Build Tools** from  
>    https://visualstudio.microsoft.com/visual-cpp-build-tools/  
>    (select "Desktop development with C++")
>
> 2. Then retry:

>    ```bash
>    pip install dlib
>    pip install face-recognition
>    ```
Your problem is Python 3.14 — it's too new, and dlib doesn't have pre-built wheels for it. The fix is simple: use Python 3.11 alongside it.

Let me check your current setup first:

Yeah — Python 3.14.3. dlib won't build for this. Here's the fastest fix:


Step-by-step: Install Python 3.11 and set up a virtual environment
Step 1 — Download Python 3.11


Generating.

>
> 3. Alternatively, use a pre-compiled dlib wheel:
>    ```bash
>    pip install dlib-19.24.1-cp311-cp311-win_amd64.whl
>    ```
>    (Search for pre-built wheels at https://github.com/z-mahmud22/Dlib_Windows_Python3.x)

### Step 3 — Add Student Face Images

Place clear, front-facing photos of each student in the `known_faces/` folder:

```
known_faces/
├── Sunil_Dehru.jpg        →  Recognised as "Sunil Dehru"
├── Amit_Kumar.png         →  Recognised as "Amit Kumar"
├── Priya_Sharma.jpeg      →  Recognised as "Priya Sharma"
```

**Image guidelines for best accuracy:**
- Use a clear, well-lit, front-facing photo
- One face per image
- Supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`
- The filename (without extension) becomes the student name
- Use underscores `_` for spaces (e.g., `First_Last.jpg`)

---

## 🚀 Usage

```bash
python main.py
```

### How to Use

1. **Launch** — The app opens with a live camera feed in the centre panel.
2. **Verify** — The left sidebar shows all enrolled students loaded from `known_faces/`.
3. **Detect** — Faces in the camera are automatically highlighted:
   - 🟢 **Green** = Recognised student (name displayed)
   - 🔴 **Red** = Unknown person
4. **Mark** — Click **"✅ Mark Attendance"** to log all recognised faces.
5. **Review** — The right panel shows today's attendance log in real-time.
6. **Export** — Click **"📄 Export CSV"** to confirm the CSV file location.

### Keyboard / Controls

| Control | Action |
|---|---|
| `✅ Mark Attendance` | Log recognised faces to CSV |
| `⏸ Pause / ▶ Resume` | Toggle camera feed |
| `⟳ Reload Faces` | Re-scan `known_faces/` without restarting |
| `📄 Export CSV` | Show path to attendance CSV |
| Close window | Safely releases camera and exits |

---

## 📊 Attendance CSV Format

The `daily_attendance.csv` file is auto-generated with this structure:

| Name | Date | Time |
|---|---|---|
| Sunil Dehru | 2026-04-23 | 09:15:32 |
| Amit Kumar | 2026-04-23 | 09:16:05 |

---

## ⚙️ Configuration

You can tweak these constants at the top of `main.py`:

| Variable | Default | Description |
|---|---|---|
| `KNOWN_FACES_DIR` | `"known_faces"` | Folder containing student images |
| `ATTENDANCE_FILE` | `"daily_attendance.csv"` | Output CSV path |
| `CAMERA_INDEX` | `0` | Camera device index (0 = default webcam) |
| `FACE_MATCH_TOLERANCE` | `0.50` | Lower = stricter matching (range: 0.0–1.0) |
| `FRAME_RESIZE_FACTOR` | `0.25` | Downscale factor for faster processing |
| `PROCESS_EVERY_N_FRAMES` | `3` | Run detection every Nth frame |

---

## 🧪 Tech Stack

| Technology | Purpose |
|---|---|
| **Python 3.10+** | Core language |
| **CustomTkinter** | Modern dark-themed GUI framework |
| **OpenCV** (`cv2`) | Webcam capture and image processing |
| **face_recognition** | Face detection and identity matching (uses dlib) |
| **NumPy** | Numerical operations for encoding comparison |
| **Pillow** | Image format conversion for Tkinter display |

---

## 📝 License

This project was built as a college lab assignment. Free to use for educational purposes.
