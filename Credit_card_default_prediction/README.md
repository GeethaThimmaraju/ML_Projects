# Credit Card Default Prediction

## 🔍 Objective
Build a machine learning model to predict whether a customer will default on their credit card payment next month using historical credit data.

## 📊 Dataset
- **Source:** UCI Machine Learning Repository
- **File:** `default of credit card clients.csv`
- Contains data from 30,000 credit card clients in Taiwan.

## 💡 Features Used
- Demographic features (education, marriage, age)
- Credit data (limit, bill statement, previous payments)
- Target variable: `default payment next month`

## 🧠 Models Implemented
- Logistic Regression (with `class_weight='balanced'`)
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM with `class_weight='balanced'`)

## ⚙️ Preprocessing
- Dropped `ID` column
- Converted all features to numeric
- Scaled features using `StandardScaler`
- Split dataset into training and testing (80/20)

## 📈 Evaluation Metrics
- Accuracy
- Precision, Recall, F1-Score
- Confusion Matrix (visualized)

## ✅ Results Summary
| Model               | Accuracy | Notes                          |
|--------------------|----------|-------------------------------|
| Logistic Regression| ~82%     | Class-balanced                |
| KNN                | ~79%     | Basic model                   |
| SVM                | ~83%     | Best performer with scaling   |

## 🗂 Project Structure
```
credit_card_default_prediction/
├── credit_card_default_prediction.ipynb
├── default of credit card clients.csv
├── README.md
└── requirements.txt
```

## 🚀 How to Run
1. Install packages from `requirements.txt`
2. Run `credit_card_default_prediction.ipynb` in Jupyter Notebook
3. All results will display in notebook outputs

## 📌 Author
Created by [Your Name] as part of machine learning portfolio.
