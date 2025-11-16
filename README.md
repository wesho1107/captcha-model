# CAPTCHA Recognition System – Repository Overview (Group 37)

This repository contains the code and poster for our **CAPTCHA Recognition System**

1. **Data Cleaning & Preprocessing** - Remove noise, fix labels, and segment CAPTCHAs into single‑character images.  
2. **CNN‑based Recognition** – Train multiple CNN architectures.
3. **Data Generation with GAN** – Generate synthetic CAPTCHA images.

---

## Repository Structure

```text
captcha_model/
├── backend/
├── cleaned_data/
├── frontend/
├── jiaen/
├── khoonsun/
├── owen/
├── rcnn/
├── Yuhao/
├── .gitignore
├── CS4243_Poster_Group_37.pdf
└── README.md
```

### `backend/` and `frontend/`
Contain the code for a simple program/UI that runs the CAPTCHA model during inference and outputs the predicted string.

### `cleaned_data/`
Contains ZIP folders of all cleaned datasets produced during our image-cleaning process.

### `jiaen/`
Contains:
- CTC-based OCR - This was explored but is **not used** in the final pipeline.
- E2CNN - Used in the final pipeline

### `khoonsun/`
Contains the K-means colour-segmentation code - This was explored but is **not used** in the final pipeline.

### `owen/`
Contains the GAN-based CAPTCHA generator used to produce synthetic training data.

### `rcnn/`
Contains the Faster R-CNN approach - This was explored but is **not used** in the final pipeline.

### `yuhao/`
Contains the main implementation:
- Image preprocessing  
- DBSCAN-based segmentation  
- CNN models - Used in the final pipeline

---

## Contributions:
- Tu Jia En: Data Cleaning, CNN, OCR-style with BiLSTM CTC
- Guo Yuhao: Data Cleaning, Preprocessing, Segmentation (DBSCAN), SqueezeNet, CNN
- Ong Sheng Jin Owen: Data Cleaning, GAN
- Ho Jia Cheng: Data Cleaning, Segmentation, FasterRCNN, Inference Visualisation
- Yek Khoon Sun: Data Cleaning, Preprocessing, Segmentation (K-clustering)