# Real-Time Classroom Attendance Tracker

A modular, robust, real-time classroom attendance tracking system powered by Computer Vision. The system leverages OpenCV Haar Cascades for face detection and Local Binary Patterns Histograms (LBPH) for face recognition. It features custom exit zone monitoring, a grace-period exit window, live terminal dashboard reporting, and automated CSV attendance logging for both students and teachers.

---

## 🚀 Key Features

* **Webcam & Video File Enrollment:** Capture face frames for student or teacher enrollment from a live webcam feed or raw video files.
* **Frontal & Profile Face Cascade Detection:** Utilizes both frontal and side profile cascade classifiers to detect faces at multiple angles.
* **Incremental Model Training:** OpenCV LBPH Face Recognizer loads, updates, and persists model states without losing previously trained faces.
* **Custom Exit Zone Setup:** Admin interface to dynamically select and draw a door/exit rectangle directly on the camera viewport.
* **Grace-Period Disappearance Engine:** Intelligent tracking that allows students a configurable grace period before marking them OUT if they disappear after being spotted in the exit zone.
* **Live CLI Dashboard:** Real-time console status dashboard displaying active students, current status (IN, OUT, PENDING), and current presence percentages.
* **Automated CSV Reports:** Generates fully detailed student attendance logs and dedicated teacher arrival logs under the `reports/` directory.

---

## 📂 Project Directory Structure

```text
├── main.py                     # Primary entry point & main terminal menu
├── requirements.txt            # Python external library dependencies
├── README.md                   # Setup guide and instructions
├── config/
│   ├── haarcascade_frontalface_default.xml   # OpenCV Frontal Face Haar Cascade
│   ├── haarcascade_profileface.xml           # OpenCV Profile Face Haar Cascade
│   └── exit_zone.json                        # Saved custom exit zone coordinates
├── data/
│   ├── faces/                  # Temporary enrollment raw photos (cleared post-training)
│   ├── models/
│   │   ├── lbph_model.yml      # Trained LBPH Face Recognition Model
│   │   └── label_map.json      # Enrollment ID, name, and role metadata
│   └── videos/                 # Input folder for batch video enrollment
├── modules/
│   ├── camera.py               # Handles camera initialization and optimal resolution
│   ├── dashboard.py            # Renders the live CLI dashboard updates
│   ├── detector.py             # Basic test execution interface for face recognition
│   ├── enroll.py               # Frontal/Profile face collection and dataset prep
│   ├── exit_zone.py            # Dynamic door region selection and point checking
│   ├── presence.py             # Formulates final duration presence percentages
│   ├── reporter.py             # Generates attendance logs and teacher reports in CSV
│   ├── tracker.py              # Presence state machine (IN / OUT / PENDING / Grace checks)
│   └── train.py                # Performs stratified splits and LBPH recognizer updates
└── reports/
    └── *.csv                   # Automated CSV attendance sheets
```

---

## 🛠️ Installation & Environment Setup

Follow these steps to configure your environment and run the classroom tracker.

### Step 1: Clone the Repository
Clone the project directory to your local machine and navigate into the folder:
```powershell
git clone <repository-url>
cd "Computer Vision/PROJECT MAIN"
```

### Step 2: Create a Python Virtual Environment
Keep your dependencies isolated. Create a local virtual environment:
```powershell
# Windows & macOS/Linux
python -m venv venv
```

### Step 3: Activate the Virtual Environment
Activate your environment based on your current OS and terminal:
* **Windows (PowerShell):**
  ```powershell
  .\venv\Scripts\Activate.ps1
  ```
* **Windows (Command Prompt):**
  ```cmd
  .\venv\Scripts\activate.bat
  ```
* **macOS / Linux (Bash/Zsh):**
  ```bash
  source venv/bin/activate
  ```

### Step 4: Install Dependencies
Install all required libraries (OpenCV Contrib and NumPy) directly from the `requirements.txt` file:
```powershell
pip install -r requirements.txt
```
> **Note:** The `opencv-contrib-python` library is explicitly used instead of `opencv-python` to provide the native face recognition contrib modules (`cv2.face.LBPHFaceRecognizer_create`).

---

## 🎯 Step-by-Step Workflow & Commands

Run the main dashboard interface from the terminal:
```powershell
python main.py
```

### Workflow Steps:

#### 1. Enroll a Person (Option `1`)
Captures raw face frames from a webcam or a pre-recorded video file.
* Select **Option 1**.
* Choose your input source: **(1) Webcam** or **(2) Video file** (placed in `data/videos/`).
* Enter the registration number (ID), full name, and role (`student` / `teacher`).
* The system will collect 300 processed frames across different angles (Front, Left, Right, Up, Down) and save them inside `data/faces/<ID>`.

#### 2. Train the Face Recognition Model (Option `2`)
Compiles the dataset and trains the LBPH model:
* Select **Option 2**.
* The training script loads the faces, performs a stratified **70/15/15** Train/Validation/Test split, updates the model incrementally without discarding old data, evaluates performance, writes/updates `data/models/lbph_model.yml`, and deletes the raw photos to free disk space.

#### 3. Setup the Exit Zone (Option `3`)
Define where the exit door is located to trigger tracking:
* Select **Option 3**.
* A frame window will open. Click and drag your mouse cursor to draw a box/rectangle over the classroom door or exit path.
* Press **ENTER** or **SPACE** to lock in the coordinates and save the configuration to `config/exit_zone.json`.

#### 4. Start the Tracker Session (Option `4`)
Run the real-time tracking session:
* Select **Option 4**.
* Select the active teacher and enter the class section name and class duration in seconds.
* The webcam feed will open. The system will start tracking student entry times, exit warnings, grace periods, and compute live presence metrics on the terminal dashboard.
* Press **Q** in the video window to stop the session manually. The session will automatically save final CSV attendance reports in the `reports/` folder.
