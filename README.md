# Credit Card Spender Classifier 💳

A machine learning project to classify credit card users as **High Spenders** or **Low Spenders** using financial behavioral data.

---

## 🧠 Problem Statement

Banks and financial institutions often segment their customers to identify high-value clients. This project builds a model to classify credit card users based on features such as average purchases, balance, and credit limit.

---

## 📊 Dataset

- The dataset contains anonymized credit card usage patterns.
- Features include: `Balance`, `AvgPurchase`, `CreditLimit`, `Payments`, etc.

---

## ⚙️ Methodology

- **Language/Tools**: Python, Pandas, scikit-learn, Matplotlib
- **Model Used**: Random Forest Classifier
- **Steps**:
  - Data Cleaning
  - Feature Engineering (`HighSpender` column based on median spending)
  - Model Training using train-test split
  - Evaluation: Accuracy, Confusion Matrix, Classification Report
  - Feature Importance Plot

---

## 📈 Results

- Accuracy: ~85%
- Top predictive features:
  - `AvgPurchase`
  - `CreditLimit`
  - `Balance`
- Visualized feature importance using bar chart

---

## 📌 Key Learnings

- Random Forest performs well for binary classification
- Feature importance helps in interpreting economic behavior
- Great foundation for segmentation-based decision-making in policy and business

---

## 🔗 Project Link

[GitHub Repo](https://github.com/SuryanshShukla338/CreditCard-Spender-Classifier)

---

## 🧠 Future Scope

- Try Logistic Regression and compare performance
- Deploy using Streamlit for live demo
- Extend for multi-class segmentation

---

## ✨ Author

**Suryansh Shukla**  
M.Sc. Economics | Data & Policy Enthusiast  
[LinkedIn](https://www.linkedin.com/in/suryansh-shukla)  
