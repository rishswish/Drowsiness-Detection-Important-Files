# 🛑 Driver Drowsiness Detection System

A real-time driver monitoring system that uses deep learning and computer vision to detect signs of fatigue—such as prolonged eye closure and frequent yawning—and alert the driver to prevent potential accidents. Built with custom-trained YOLOv5 models and video stream processing.

---

## 🚗 Motivation

Drowsy driving is a major cause of traffic accidents worldwide. Detecting early signs of fatigue—like eye closure or yawning—can help reduce the risk of crashes. This project builds a real-time safety system that monitors a driver’s alertness using a webcam and deep learning models.

---

## 📷 System Overview

The system continuously analyzes live video feed and performs:

1. **Face Detection & Alignment**
2. **Eye and Mouth Region Extraction**
3. **Drowsiness Detection using YOLOv5 models**
4. **Voice Alarm Triggering**

If either of the following is detected:
- Eyes closed for more than a threshold duration
- Frequent yawning within a short time window

An **audio alert** is triggered to wake the driver.

---

## 🛠️ Tools & Technologies

| Component             | Technology/Library     |
|----------------------|------------------------|
| Object Detection      | YOLOv5 (PyTorch)       |
| Image Processing      | OpenCV, NumPy          |
| Model Training        | Custom-labeled dataset |
| Preprocessing         | Face Alignment, Gaussian Blur |
| GUI/Alerts            | PyGame (for sound)     |

---

## 🧪 Dataset

- Custom dataset of **1,200+ annotated images**
- Includes:
  - Open/closed eyes
  - Yawning/non-yawning faces
- Augmented with:
  - Lighting variations
  - Head pose angles
  - Gaussian blur

---



## Demo Video

https://github.com/rishswish/Drowsiness-Detection-Important-Files/assets/89961075/9b276f34-9ccf-4a89-a863-5ee4e56af88b


https://github.com/rishswish/Drowsiness-Detection-Important-Files/assets/89961075/f99ee77a-72b9-4f50-baec-1fef160f5ff6

