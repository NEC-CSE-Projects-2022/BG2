
# BG2 – INTELLIGENT PEST CLASSIFICATION AND REMEDY RECOMMENDATION SYSTEM USING VISION TRANSFORMERS AND DOMAIN-AWARE SUMMARIZATION

## Team Info
- **22471A0580 — Satwika Jahnavi Chakka** ( [LinkedIn](https://linkedin.com/in/satwikajahnavi) )
_Work Done: Model training, evaluation, preprocessing_

- 22471A0590 — **Thrisony Gayam** ( [LinkedIn](https://linkedin.com/in/xxxxxxxxxx) )
_Work Done:documentation, Dataset preparation_

- 22471A05A1 — **Jahnavi Kappa** ( [LinkedIn](https://linkedin.com/in/xxxxxxxxxx) )
_Work Done: Research analysis, testing_



---

## Abstract
Farmers face major crop losses due to delayed or inaccurate pest identification, leading to excessive pesticide usage and reduced productivity. This project proposes an AI-powered pest detection system using Vision Transformers (ViT) that classifies rice pests from images and provides domain-aware treatment and prevention recommendations. The system acts as a digital agricultural assistant, converting complex pest data into simple, actionable guidance for farmers, students, and agricultural workers.

---

## Paper Reference (Inspiration)
👉 **FasterPest: A Multi-Task Classification Model for Rice Pest Recognition
  – Author Names Xiaoyun Zhan; Cong Zhang; Zheng Wang; Yuantao Han; Peng Xiong; Linfeng He
 ]((https://ieeexplore.ieee.org/document/10734263))**
Original conference/IEEE paper used as inspiration for the model.

---

## Our Improvement Over Existing Paper
Unlike traditional pest classifiers that only output class labels, our system integrates a domain-aware recommendation layer that provides pesticide suggestions, severity analysis, and prevention tips. The focus is not just prediction accuracy, but real-world usability and decision support for farmers.

---

## About the Project
**What it does**
      Takes a crop image and predicts the pest type using deep learning, then provides treatment and prevention guidance.

**Why it is useful**
      Reduces crop losses
      Minimizes pesticide misuse
      Provides expert-level advice instantly

**Workflow**
Image → Preprocessing → ViT Model → Pest Prediction → Knowledge Base → Recommendations

---

## Dataset Used
👉 **IP102 Pest Dataset
[(Dataset URL)](https://github.com/xpwu95/IP102)**

**Dataset Details:**
    40 pest classes used
    75,000+ real-world images
    Diverse lighting, backgrounds, and pest conditions

---

## Dependencies Used
    Python
    PyTorch
    timm
    OpenCV
    NumPy
    Pandas
    Albumentations
    Matplotlib
    Seaborn

---

## EDA & Preprocessing
    Removed corrupted images
    Resized all images to 224×224
    Normalized pixel values
    Applied augmentation (flip, rotate, blur, brightness)
    Balanced class distribution

---

## Model Training Info
    Base Model: vit_base_patch16_224
    Optimizer: AdamW
    Loss Function: CrossEntropyLoss
    Epochs: 20
    Learning Rate: 2e-5

---

## Model Testing / Evaluation
    Accuracy: 96%
    Precision: 95.4%
    Recall: 94.7%
    F1 Score: 95.0%
    Confusion matrix generated for all 40 classes

---

## Results
The model successfully classifies visually similar pests and provides accurate treatment suggestions. Compared to CNN-based systems, ViT shows superior generalization and robustness under real-world conditions.

---

## Limitations & Future Work
    Currently limited to rice pests
    No real-time mobile deployment yet

**Future scope:**

    Multi-crop support
    Regional language interface
    Drone-based pest monitoring
    Cloud API deployment
---

## Deployment Info

The system is currently deployed as a live prototype on Hugging Face Spaces, enabling real-time pest image classification and recommendation through a web-based interface. Users can upload crop images and receive instant AI-driven predictions along with treatment and prevention guidance. The deployment demonstrates practical usability in an online environment, while future plans include optimizing performance for mobile platforms, integrating cloud APIs, and supporting large-scale user access.
---
