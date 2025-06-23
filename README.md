# 🏦 Loan Approval Prediction Project

This project predicts whether a loan will be approved or not based on applicant details using Machine Learning.

---

## 📌 Overview

This is a classification project using **Logistic Regression** to predict loan approval status.  
The model is trained on historical loan applicant data provided in a CSV file.

---

## 📁 Dataset

- File: `loan_prediction.csv`
- Rows: 614
- Features include: Gender, Marital Status, Income, Education, Credit History, Property Area, etc.

---

## ⚙️ Technologies Used

| Tool/Library      | Purpose                          |
|-------------------|----------------------------------|
| Python            | Programming language             |
| Pandas            | Data manipulation & analysis     |
| NumPy             | Numerical operations             |
| scikit-learn      | Machine Learning model           |
| Joblib            | Saving the model                 |
| Seaborn & Matplotlib | Data visualization            |
| VS Code           | Code editor                      |

---

## 📊 Exploratory Data Analysis (EDA)

Key graphs include:

- Loan approval status count
![Loan Approval Status](https://github.com/user-attachments/assets/7e66e7ef-49c8-40fa-972f-7fc95c476f0b)

- Applicant income distribution
![Applicant Income](https://github.com/user-attachments/assets/ac77cb66-a497-4bd6-abbc-ff35d4e077f1)

- Loan amount by education level
![Loan amount vs Education](https://github.com/user-attachments/assets/ff35cf98-782e-48ee-a886-e2713e933ce2)

- Property area vs loan status
![Property area vs Loan Status](https://github.com/user-attachments/assets/b25a5000-74f1-4069-9368-189dbff76c1e)
 
- Correlation heatmap
![Correaltion  Heatmap](https://github.com/user-attachments/assets/e9521922-80de-4386-94c2-2e264cc8c5fb)


## 🧠 Model Details

- **Algorithm Used**: Logistic Regression
- **Accuracy**: ~78.8%
- **Model File**: `loan_approval_model.pkl`

---
## 🚀 Installation & Usage
git clone https://github.com/your-username/loan-approval-prediction.git

cd loan-approval-prediction

pip install -r requirements.txt

python loan_approval.py




