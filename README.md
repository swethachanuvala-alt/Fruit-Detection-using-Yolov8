# 🍎 Fruit Object Detection using YOLOv8

## 📌 Project Overview

Fruit Object Detection is a computer vision project that detects and identifies different types of fruits in an uploaded image using the YOLOv8 object detection model. The application processes images, draws bounding boxes around detected fruits, and displays the detected fruit names along with confidence scores through a simple Streamlit web interface.

---

## 🚀 Features

* Detects multiple fruits in a single image.
* Displays bounding boxes around detected fruits.
* Shows the predicted fruit name.
* Displays confidence scores for each detection.
* User-friendly Streamlit interface.
* Supports JPG, JPEG, and PNG images.

---

## 🛠️ Technologies Used

* Python
* YOLOv8 (Ultralytics)
* OpenCV
* PyTorch
* Streamlit
* NumPy
* Pillow

---

## 📂 Dataset

* Dataset Source: Kaggle (Roboflow Fruit Detection Dataset)
* Dataset Format: YOLO Object Detection
* Images are organized into:

  * Train
  * Validation
  * Test
* Each image has a corresponding YOLO annotation (.txt) file.

---

## ⚙️ Model Training

* Model: YOLOv8 Nano (`yolov8n.pt`)
* Custom-trained on the fruit detection dataset.
* Evaluated using YOLO validation metrics such as:

  * Precision
  * Recall
  * mAP@0.5
  * mAP@0.5:0.95

---

## 💻 Application Workflow

1. Upload a fruit image.
2. The trained YOLOv8 model analyzes the image.
3. Fruits are detected and localized with bounding boxes.
4. The application displays:

   * Detected fruit names
   * Confidence scores
   * Annotated output image

---

## 📁 Project Structure

```text
CV/
│── app.py
│── best.pt
│── requirements.txt
│── README.md
```

---

## ▶️ Installation

Clone the repository:

```bash
git clone <repository-url>
cd CV
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment:

Windows

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

## 📸 Output

The application displays:

* Uploaded image
* Detected objects
* Bounding boxes
* Confidence scores
* Predicted fruit names

---

## 🔮 Future Enhancements

* Real-time webcam detection.
* Video object detection.
* Mobile-friendly deployment.
* Detection statistics dashboard.
* Confidence threshold adjustment.

---

## 👩‍💻 Author

**Swetha**

AI & Machine Learning Enthusiast

Interested in Computer Vision, Deep Learning, Data Science, and AI-powered applications.
