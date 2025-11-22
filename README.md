# 📩 SMS Spam Detection using Machine Learning (Ensemble Learning)

This project focuses on classifying SMS messages as **Spam** or **Ham** using machine learning and NLP techniques. Using the **spam.csv** dataset, the project implements text preprocessing, TF-IDF feature extraction, multiple ML algorithms, and ensemble learning for high-accuracy spam detection. It connects all major ML modules such as Regression, SVM, Tree Models, Boosting, PCA, Model Selection, and more.

---

## 📂 Dataset Used
- **spam.csv**
  - label — spam/ham
  - message — SMS text content

---

## 🚀 ML Concepts Covered (Mapped to Course Modules)

### **1. Advanced Regression**
- Underfitting vs overfitting
- Regularization concepts (L1/L2)
- Feature selection techniques

### **2. Support Vector Machine (SVM)**
- Applied SVM classifier for spam detection
- Margin maximization & kernel tricks

### **3. Tree Models**
- Decision Tree
- Random Forest
- Feature importance understanding

### **4. Boosting Algorithms**
- XGBoost
- AdaBoost / Gradient Boosting
- Weak learners → strong learners

### **5. Model Selection (Practical)**
- Cross-validation
- Metric comparison
- Selecting best model using Ensemble Voting

### **6. PCA (Optional Step)**
- Dimensionality reduction on TF-IDF vectors
- Visualizing text data patterns

### **7. Clustering (Module Concept)**
- Understanding K-Means / Hierarchical Clustering
- Not applied directly but part of overall ML learning

### **8. Business Case Study Approach**
- Complete pipeline: preprocessing → feature extraction → modeling → evaluation

---

## 🛠️ Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- NLTK
- XGBoost
- Matplotlib / Seaborn

---

## ⚙️ Project Workflow
1. Load **spam.csv**
2. Data cleaning & text preprocessing  
3. Convert text to vectors using **TF-IDF**
4. Train models:
   - Logistic Regression
   - SVM
   - Decision Tree
   - Random Forest
   - XGBoost
5. Build **Ensemble Voting Classifier**
6. Evaluate using Accuracy, Precision, Recall, F1-Score

---

## 📊 Model Evaluation
- Accuracy  
- Precision  
- Recall  
- F1-Score  
- Confusion Matrix  

Expected performance: **97% – 99% accuracy**

---

## 📁 Project Structure
SMS-Spam-Detection/
│-- spam.csv
│-- spam_detection.ipynb
│-- README.md
└── models/

---

## 🎯 Outcome
- High-accuracy SMS spam classifier  
- Complete ML + NLP pipeline  
- Concepts applied from Regression, SVM, Tree Models, Boosting, PCA, Model Selection  
- Strong practical understanding of applied machine learning  

---

## 🚀 Future Improvements
- Deploy with Flask / FastAPI
- Add LSTM / BERT deep learning model
- Build a real-time spam detection app

