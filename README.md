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

## 2. Install Python Dependencies

Install required Python libraries:

```bash
pip install -r requirements.txt
```

---

## 3. Install DVC (Google Drive Support)

This project uses **DVC** to manage large datasets.

```bash
pip install dvc[gdrive]
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
