# Chicken Counting Using Object Detection

This project focuses on **chicken counting** from images using object detection techniques.
A **pure SVM-based approach** is implemented first, then compared with **YOLO**.

---

## 1. Clone the Repository

```bash
git clone https://github.com/Duc-Vu/chicken_couting_using_object_detection.git
cd chicken_couting_using_object_detection
```

---

## 2. Install Python Envirovment

```bash
conda env create -f environment.yml
conda activate chicken_counting
```

---

## 3. Install DVC

This project uses **DVC** to manage large datasets.

```bash
pip install dvc[s3]
```

Check installation:

```bash
dvc --version
```

---

## 4. Download Dataset with DVC

Download the dataset from the DVC remote storage:

```bash
dvc pull
```

After this step, the `data/` directory will be downloaded automatically.

## 5. About Dataset

https://universe.roboflow.com/uit-6vkfy/chicken-j6niq/dataset/6
