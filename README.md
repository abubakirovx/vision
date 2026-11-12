# AI Vision Assistant

### Real-Time Object Detection & Multimodal Visual Intelligence

A privacy-focused, locally powered AI application that combines real-time computer vision with multimodal language understanding.

AI Vision Assistant transforms a webcam into an interactive visual intelligence system. It detects objects in real time and allows users to ask natural-language questions about their surroundings.

Built with **Python, FastAPI, OpenCV, YOLO11, Ollama, Qwen2.5-VL, React, and TypeScript**.

---

## Overview

AI Vision Assistant integrates two complementary AI capabilities:

**Object Detection:** YOLO11 analyzes webcam frames to identify supported objects, draw bounding boxes, and calculate detection statistics.

**Visual Understanding:** Qwen2.5-VL interprets camera images and answers natural-language questions about visible objects, scenes, and activities.

Unlike applications that require paid cloud-based AI services, the system runs locally using pretrained models.

### Key Highlights

- Real-time webcam object detection
- Multimodal visual question answering
- Local AI inference through Ollama
- Interactive camera and AI conversation interface
- Object counting and confidence metrics
- English and Uzbek question support
- Privacy-focused processing
- Modern responsive web interface
- No paid AI API required

---

## Technology Stack

| Layer | Technologies |
|---|---|
| Frontend | React, TypeScript, Vite, Tailwind CSS |
| Backend | Python 3.11, FastAPI |
| Computer Vision | OpenCV, Ultralytics YOLO11n |
| Vision-Language Model | Qwen2.5-VL 3B |
| Model Runtime | Ollama |
| Communication | REST API |
| Environment | Local development |

---

## System Architecture

```text
                  USER
                   |
                   v
           React Web Interface
                   |
              Browser Camera
                   |
             Captured Frames
                   |
                   v
              FastAPI Backend
                   |
          +--------+--------+
          |                 |
          v                 v
      YOLO11n          Qwen2.5-VL
          |                 |
    Object Detection   Visual Reasoning
          |                 |
          v                 v
    Bounding Boxes     AI Responses
          |                 |
          +--------+--------+
                   |
                   v
            React Dashboard
```

### Processing Flow

1. The user activates the webcam.
2. The browser captures frames.
3. FastAPI receives image data.
4. YOLO11 performs object detection.
5. Detection results are displayed on the camera feed.
6. The user submits a question about the scene.
7. Qwen2.5-VL analyzes the attached image.
8. The generated response appears in the conversation panel.

---

## Core Features

### 1. Real-Time Object Detection

YOLO11n detects objects belonging to its supported categories.

The application displays:

- Object labels
- Bounding boxes
- Confidence scores
- Total detected objects
- Unique object classes
- Detection latency

### 2. Multimodal AI Assistant

Users can ask questions such as:

- "What objects can you see?"
- "Describe my surroundings."
- "How many people are visible?"
- "What is happening in this scene?"
- "What color is the object I'm holding?"

Qwen2.5-VL analyzes the current camera image and generates a contextual response.

### 3. Local AI Processing

The application uses Ollama to run Qwen2.5-VL locally.

Benefits include:

- No paid inference API
- No cloud AI dependency
- Local image processing
- Greater control over model execution

Camera frames are processed locally by the application rather than sent to a hosted AI provider.

### 4. Modern Web Interface

The frontend provides:

- Live camera preview
- Object detection overlays
- Interactive AI conversation
- Model availability indicators
- Detection performance metrics
- Loading and error states
- Responsive layouts

---

## Installation

### Prerequisites

Install the following:

- Python 3.11
- Node.js and npm
- Ollama
- Git
- A webcam

### 1. Clone the Repository

```bash
git clone https://github.com/abubakirovx/vision.git
cd Ai_assistance
```

### 2. Install the Vision Model

Install Ollama and download the vision-language model:

```bash
ollama pull qwen2.5vl:3b
```

Verify installation:

```bash
ollama list
```

### 3. Set Up the Backend

Create a Python virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install backend dependencies using the repository's dependency file.

### 4. Start FastAPI

From the project root:

```bash
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Backend health endpoint:

```text
http://127.0.0.1:8000/health
```

### 5. Start the Frontend

Open another terminal and navigate to the frontend directory:

```bash
npm install
npm run dev
```

Run these commands from the directory containing the frontend's `package.json`.

Open the local address displayed by Vite.

---

## API Overview

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Backend health check |
| POST | `/api/detect` | Detect objects in an image |
| POST | `/api/ask` | Answer questions about an image |

---

## Testing

The application includes automated backend tests and frontend build checks.

The latest reported verification included:

- 28 passing backend tests
- 12 successful real-image Q&A requests
- Concurrent request handling checks
- Successful YOLO endpoint verification

These results represent development testing, not a guarantee of production reliability.

---

## Limitations

- YOLO11n recognizes only its supported object categories.
- Small-object detection depends on image quality, distance, and lighting.
- Vision-language responses may contain inaccuracies.
- Uzbek-language responses may be less consistent than English.
- Inference performance depends on available hardware.
- Ollama inference may occasionally require recovery after runtime errors.

---

## Privacy

AI Vision Assistant is designed for local execution.

- No paid cloud AI services are required.
- Camera access is controlled by the browser.
- Model inference runs on the user's machine.
- The application does not require a user account.

Users should review their local configuration and logs before processing sensitive images.

---

## Future Improvements

Potential extensions include:

- Improved small-object recognition
- Additional vision-language models
- Voice-based interaction
- Advanced scene analysis
- Performance benchmarking
- GPU inference optimization

---

## Project Status

**Functional local prototype**

The project demonstrates the integration of real-time computer vision, multimodal AI inference, REST APIs, and a modern web interface.

It is intended as an AI engineering portfolio project and a foundation for further development.

---

## License

A license has not yet been specified.

---

**AI Vision Assistant — See. Understand. Interact.**
