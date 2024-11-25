# Breast Cancer Detection Using Hybrid Deep Learning Models

This project focuses on improving breast cancer detection by leveraging hybrid deep learning architectures, including Convolutional Neural Networks (CNNs) and Vision Transformers (ViT). Our approach aims to enhance detection accuracy and minimize false negatives, ultimately supporting early diagnosis and improved patient outcomes.

---

## 🚀 Features
- **Hybrid Models**: Combines CNNs (e.g., Xception) with Vision Transformers for robust image classification.
- **Data Augmentation**: Employs techniques like rotation, vertical flip, and zoom to improve model generalizability.
- **High Accuracy**: Achieved 96.8% accuracy and 97% recall on the CBIS-DDSM dataset and 93.23% accuracy on the IDC dataset.
- **Multiple Datasets**: Utilized CBIS-DDSM (mammograms) and IDC (histopathology images) for comprehensive evaluation.
- **Evaluation Metrics**: Assessed models using accuracy, precision, recall, F1 score, and confusion matrix.

---

## 📂 Datasets
- **CBIS-DDSM**: Mammogram images curated with segmentation maps, bounding boxes, and ROI annotations.
- **IDC Dataset**: Histopathology images with 277,524 image patches labeled as IDC positive or negative.

---

## 🛠️ Methodology
1. **Preprocessing**: Resized images, applied normalization, and augmented data for better generalization.
2. **Proposed Framework**:
   - CNN-based models for feature extraction and classification.
   - Hybrid architectures combining Xception and EfficientNetB0 with Vision Transformers.
   - Pretrained Prov-GigaPath for large-scale data processing.
3. **Evaluation**: Measured performance using confusion matrices, ROC curves, and evaluation metrics.

---

## 🧪 Results
- **CBIS-DDSM Dataset**:
  - Best Model: **Xception + Vision Transformer**
  - Accuracy: **96.8%**, Precision: **96%**, Recall: **97%**, F1-Score: **96%**
- **IDC Dataset**:
  - Best Model: **Prov-GigaPath**
  - Accuracy: **93.23%**, Precision: **90.42%**, Recall: **85.15%**, F1-Score: **87.71%**

---

## 💻 Technology Stack
- **Programming Language**: Python
- **Libraries**: TensorFlow, PyTorch, NumPy, Matplotlib
- **Models**: Xception, EfficientNetB0, Prov-GigaPath, Vision Transformer

---

## 📊 Visualizations
### CBIS-DDSM Results
- Accuracy and loss curves
- Confusion matrix

### IDC Results
- Accuracy and loss curves
- Confusion matrix

---

## 📈 Future Work
- Integrating CBIS-DDSM and IDC datasets for enhanced data diversity.
- Exploring advanced augmentation and preprocessing techniques.
- Optimizing hyperparameters for improved performance.

---

## 👩‍💻 Contributors
- **Manzi Dave Rugari** (Taylor University)  
- **Froduard Habimana** (Handong Global University)  
- **Jaeeun Lee** (Handong Global University)  
- **Djika Asmaou Houma** (Handong Global University)

---

