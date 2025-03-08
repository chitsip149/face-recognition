# **Face Authentication Using Classical Image Processing**
*A face authentication system using traditional image processing techniques.*

---

## **Overview**
This project was developed as the **final project** for the **"Introduction to Image Processing"** course at **National Dong Hwa University (NDHU)**. The goal was to implement a **face authentication system** using **classical image processing techniques** without relying on deep learning models.  

It uses **facial landmark detection, alignment, and feature normalization** to compare facial features based on **Euclidean distance**.

### **Features**
- Face alignment based on **eye landmarks**  
- Symmetric **feature adjustment** for consistent representation  
- Euclidean distance-based **face matching**  
- **Threshold optimization** to improve recognition performance  
- Performance evaluation using **accuracy, precision, recall, and F1-score**

---

## **Dataset**
The dataset consists of two main directories:

 **Face_DB/** (Authorized faces)  
- `Images/` → Frontal face images (e.g., `NAME_000.jpg`)  
- `Landmark_data/` → Corresponding **landmark CSV files**  

 **Test_DB/** (Test images)  
- `Images/` → Face images at different angles  
- `Landmark_data/` → Landmark CSVs for each test image  

Each person has **6 test images**, totaling **90 test samples**.

---

## **Methodology**
### **1️. Face Alignment Using Eye Landmarks**
- Faces were **aligned** using **eye landmarks** to ensure consistency.

### **2️. Symmetrization of Facial Features**
- Each landmark was **mirrored** and adjusted to **reduce asymmetry** caused by angle variations.

### **3️. Feature Extraction & Normalization**
- Jawline **excluded** to prevent distortions.  
- **Relative distances between landmarks** used instead of absolute positions.  
- Feature distances were **normalized** using the width of the left eye.

### **4️. Face Matching Using Euclidean Distance**
- Euclidean distance was computed between the aligned and normalized feature vectors.

### **5️. Threshold Selection for Best Performance**
- Thresholds were tested between **50 and 120** to optimize results.  
- **Best threshold: 73** (Selected based on F1-score).

---

## **Results**
| Metric       | Value  |
|-------------|--------|
| Accuracy    | 52.22% |
| Precision   | 57.69% |
| Recall      | 58.82% |
| F1-score    | 58.25% |

### **Example Predictions**
| Scenario               | Test Image        | Predicted Person | Verdict |
|------------------------|------------------|------------------|---------|
| ✅ Correct Match       | `Chloe_001.jpg`  | `Chloe`          | ✔️ Accepted |
| ❌ False Positive      | `Ethan_002.jpg`  | `Sebastian`      | ❌ Imposter Accepted |
| ❌ Wrong Identity      | `Zoe_000.jpg`    | `Chloe`          | ❌ Misclassified |
| ✅ Correct Rejection  | `Lucas_004.jpg`  | —                | ✔️ Rejected |
| ❌ False Rejection    | `Wyatt_004.jpg`  | —                | ❌ Incorrectly Rejected |

---

## **How to Run the Project**
### **1️. Clone the Repository**
```bash
git clone https://github.com/chitsip149/face-recognition.git
cd "411021365 Nguyen Minh Trang"
```

### **2️. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **3️. Run the Face Authentication System**
```bash
python face_auth.py
```

### **4️. (Optional) Run Performance Evaluation**
```bash
python evaluate.py
```

## ** Key Learnings**
- Preprocessing (alignment & normalization) is critical in face recogition.
- Feature selection (excluding jawline) improves robustness to angles.
- Euclidean distance works for simple comparisons but has limitations.
- Empirical evaluation helps fine-tune parameters like the threshold.


## **Future Improvements**
- Use machine learning models (e.g., SVM, k-NN) for improved classification.
- Apply deep learning (CNNs, OpenCV DNN, or FaceNet) for robust face recognition.
- Optimizr featue selection using principal component analysis (PCA).
- Expand dataset to test on more diverse facial variations.

## **License**
This project is licensed under the MIT License.
