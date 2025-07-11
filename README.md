# Face Recognition System

A modern, modular real-time face recognition system in Python, featuring detection, encoding, and identification using deep learning. This project is suitable for applications like real-time recognition, attendance, and access control.

## Demo

![Screenshot](assets/friends1.jpg)

[![Watch the video](https://github.com/AIApproach/Face-Recognition-System/blob/main/assets/Face%20Recognition%20System.mp4)](https://github.com/AIApproach/Face-Recognition-System/blob/main/assets/Face%20Recognition%20System.mp4)


## Features

- **Face Detection**: Uses MTCNN for accurate face localization.
- **Face Embedding**: Employs a pre-trained FaceNet model for robust face representations.
- **Identification**: Compares face embeddings with a known database using cosine similarity.
- **Real-Time Recognition**: Processes webcam (or video) streams and displays results live.
- **Attendance Logging**: Optionally records recognized faces with timestamps.
- **Flexible Visualization**: Overlays bounding boxes, names, and FPS information on video.
- **GPU Support**: Configurable memory limits for TensorFlow GPU usage.

## Getting Started

### Prerequisites

- Python 3.7+
- Recommended: GPU with CUDA support for faster inference

### Installation

1. **Clone the repository:**

   ```sh
   git clone https://github.com/AIApproach/Face-Recognition-System.git
   cd Face-Recognition-System
   ```

2. **Install requirements:**

   ```sh
   pip install -r requirements.txt
   ```

3. **Download models:**

   Place your pre-trained FaceNet model in `models/facenet/`.

### Usage

#### Real-Time Face Recognition

```sh
python real_time_face_recognition.py
```

Add `--debug` for additional output.

#### Attendance Recording

```sh
python attendance.py
```

This will log recognized names and timestamps to a CSV file.

#### Adding New Identities

You can add new faces to the system using the provided methods in `face.py` (see `Recognition.add_identity(image, person_name)`).

### Code Structure

- `face.py`: Main module for detection, embedding, and identification.
- `real_time_face_recognition.py`: Real-time demo application.
- `attendance.py`: Demo application with CSV attendance logging.
- `src/`: Supporting code for FaceNet and alignment.
- `models/`: Place your TensorFlow FaceNet model here.

### Example

Running `real_time_face_recognition.py` will open your webcam, detect faces, identify them, and display names live.

### License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

### Author

(c) 2025 Tareq Al-Kushari

---

*For questions or contributions, open an issue or pull request on GitHub.*
