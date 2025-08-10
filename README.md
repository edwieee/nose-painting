

## Basic Details
### Team Name: KINNAM


### Team Members
- Team Lead: Chrismon P Lijo - CCE
- Member 2: Edwin BIju - CCE


Project Description
A hands-free virtual mouse cursor controlled entirely by your nose. Just look at your webcam, move your head, and blink to click — because touching a mouse is overrated.

The Problem (that doesn't exist)
In this fast-paced world, lifting your hand to control a mouse is exhausting. Why waste precious wrist energy when your face can do the job?

The Solution (that nobody asked for)
We built a webcam-powered nose tracker that follows your nostrils across the screen, paired with blink-based clicking. One blink for a click, two blinks for a double click — it’s like Morse code, but with your eyelids.

Technical Details
Technologies/Components Used
For Software:

Languages used: Python

Frameworks used: None (pure OpenCV + dlib magic)

Libraries used: OpenCV, dlib, pyautogui, NumPy

Tools used: Webcam, Virtual Environment, Good Lighting

For Hardware:

Main components: Your face, a webcam, eyelids

Specifications: Any standard laptop/USB webcam (2MP+ recommended)

Tools required: Lighting setup, functioning eyelids, caffeine

Implementation
For Software:

# Installation

bash
Copy
Edit
python -m venv .venv
.venv\Scripts\activate  # Windows PowerShell
# source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
Model auto-download:
shape_predictor_68_face_landmarks.dat will be fetched from dlib.net on first run.
If it fails, download manually and place it in the project root.

# Run

bash
Copy
Edit
python main.py
Project Documentation
# Screenshots

![Screenshot1](img1.png)
Green dot proudly sitting on your nose tip — the star of the show.

![Screenshot](img2.png)
Cursor zooming across the screen, hands nowhere in sight.

# Diagrams


Webcam → Nose landmark detection → Coordinate mapping → Cursor movement → Blinking = Clicking


Project Demo
# Video
[(scrn1.mp4)](https://drive.google.com/file/d/128g9KeDfE96Nt7dY0zF6obDRYqnfms2r/view?usp=sharing)
Shows nose-guided cursor navigation and blink-based clicking in real time.



Team Contributions
Chrismon p Lijo: Lead Nose Navigator & Code Developer

Edwin Biju: Test audience for blink detection

---
Made with ❤️ at TinkerHub Useless Projects 

![Static Badge](https://img.shields.io/badge/TinkerHub-24?color=%23000000&link=https%3A%2F%2Fwww.tinkerhub.org%2F)
![Static Badge](https://img.shields.io/badge/UselessProjects--25-25?link=https%3A%2F%2Fwww.tinkerhub.org%2Fevents%2FQ2Q1TQKX6Q%2FUseless%2520Projects)



