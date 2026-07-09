# Customer Churn Prediction Dashboard


## Features
- Customer churn prediction
- Exploratory Data Analysis
- Data preprocessing pipeline
- Machine Learning model training
- Interactive Streamlit dashboard
- Business insights visualization

## Tech Stack
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- Streamlit

## Machine Learning Models
- Logistic Regression
- Random Forest Classifier

## Project Structure

## Project Structure

```
Customer_Churn/
│
├── app.py
├── data_preprocessing.py
├── train_model.py
├── eda.py
├── requirements.txt
│
├── data/
│   ├── churn.csv
│   └── cleaned_churn.csv
│
├── models/
│   ├── best_model.pkl
│   ├── scaler.pkl
│   └── model_results.csv
│
└── visuals/
    ├── churn_distribution.png
    ├── correlation_heatmap.png
    └── feature_importance_rf.png
```




## How to Run

Install dependencies:

pip install -r requirements.txt


Run dashboard:

streamlit run app.py


## Description

This project analyzes customer behavior and predicts customer churn using machine learning techniques. It helps businesses identify customers likely to leave and improve retention strategies.
