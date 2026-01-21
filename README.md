
#  Heart Disease Prediction Using Machine Learning

##  Project Overview
This project focuses on predicting the presence of heart disease using various machine learning classification algorithms. The goal is to compare different models and evaluate their performance using accuracy, confusion matrix, and classification reports.

The dataset contains medical attributes such as age, sex, blood pressure, cholesterol level, and more, which are used to determine whether a person has heart disease or not.

---

##  Dataset Information
- **Dataset Name:** Heart Disease Dataset  
- **Target Variable:** `target`
  - `0` → No Heart Disease  
  - `1` → Defective Heart  
- **Type:** Structured / Tabular Data  
- **Format:** CSV file  

### Features
- age  
- sex  
- cp (chest pain type)  
- trestbps (resting blood pressure)  
- chol (cholesterol)  
- fbs (fasting blood sugar)  
- restecg  
- thalach (maximum heart rate achieved)  
- exang (exercise induced angina)  
- oldpeak  
- slope  
- ca  
- thal  

---

##  Technologies Used
- Python  
- Pandas  
- NumPy  
- Matplotlib  
- Seaborn  
- Scikit-learn  

---

##  Project Workflow
1. Importing required libraries  
2. Loading and exploring the dataset  
3. Data preprocessing  
4. Splitting data into training and testing sets  
5. Training multiple machine learning models  
6. Evaluating model performance  

---

##  Machine Learning Models
- Logistic Regression  
- Decision Tree Classifier  
- Random Forest Classifier  
- Gaussian Naive Bayes  

---

##  Model Evaluation
Models are evaluated using:
- Accuracy Score  
- Confusion Matrix  
- Classification Report (Precision, Recall, F1-score)  

Heatmaps are used to visualize confusion matrices.

---

##  How to Run the Project

### Clone the Repository
```bash
git clone https://github.com/Anjan580/Heart-Disease-prediction-using-machine-learning
```

### Install Dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Run the Notebook
```bash
jupyter notebook Untitled.ipynb
```

---

##  Project Structure
```
├── Untitled.ipynb
├── heart_disease_data.csv
├── README.md
```

---

##  Future Enhancements
- Hyperparameter tuning  
- Feature scaling  
- Cross-validation  
- Model deployment  

---

##  Author
**Anjan Pokhrel**  
Machine Learning & Data Science Student
