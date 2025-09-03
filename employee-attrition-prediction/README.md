<<<<<<< HEAD
# 🚀 Employee Attrition Prediction with Sentiment Analysis  

This project predicts **employee attrition (turnover)** using HR data and integrates **synthetic sentiment analysis** from job-related satisfaction scores. It provides insights for HR managers through **machine learning models** and an interactive **Streamlit dashboard**.  

---

## 📌 Problem Statement  
Employee attrition poses a significant challenge to organizations, leading to increased recruitment costs, loss of knowledge, and reduced productivity. By predicting attrition early and analyzing employee sentiment, HR teams can take proactive measures to improve retention.  

---

## 🎯 Objectives  
- Predict whether an employee will leave the company.  
- Generate **synthetic sentiment scores** (Positive/Neutral/Negative) using job satisfaction features.  
- Identify **key factors influencing attrition** (e.g., overtime, income, work-life balance).  
- Build an **interactive dashboard** for HR managers.  

---

## 📊 Dataset  
Source: [Employee Attrition Prediction Dataset (Kaggle)](https://www.kaggle.com/)  

- **employee_attrition_dataset.csv** → 1,000 rows (small sample).  
- **employee_attrition_dataset_10000.csv** → 10,000 rows (large dataset).  

**Key Features**  
- Demographics: `Age`, `Gender`, `Marital_Status`  
- Job Info: `Department`, `Job_Role`, `Job_Level`, `Monthly_Income`  
- Satisfaction & Ratings: `Job_Satisfaction`, `Work_Life_Balance`, `Performance_Rating`  
- Target: `Attrition` (Yes/No)  

---

## ⚙️ Project Structure  
employee-attrition-prediction/
│
├── data/
│   ├── employee_attrition_dataset.csv
│   ├── employee_attrition_dataset_10000.csv
│
├── notebooks/
│   ├── 1_data_preprocessing.ipynb
│   ├── 2_eda.ipynb
│   ├── 3_sentiment_analysis.ipynb
│   ├── 4_model_training.ipynb
│   ├── 5_dashboard.ipynb
│
├── models/
│   ├── attrition_model.pkl
│   ├── scaler.pkl
│
├── dashboard/
│   ├── app.py   ← final Streamlit app (converted from 5_dashboard.ipynb)
│
├── requirements.txt
├── README.md


---

## 🔑 Methodology  
1. **Data Preprocessing**  
   - Handle missing values, encode categorical features, scale numerical features.  

2. **Exploratory Data Analysis (EDA)**  
   - Attrition trends by job role, department, overtime, income.  
   - Visualizations with Matplotlib, Seaborn, Plotly.  

3. **Synthetic Sentiment Analysis**  
   - Generate sentiment from `Job_Satisfaction`, `Work_Life_Balance`, `Work_Environment_Satisfaction`, `Relationship_with_Manager`.  
   - Labels: Positive / Neutral / Negative.  

4. **Feature Engineering**  
   - Create derived features like `Years_Since_Last_Promotion`, `Work_Hours_Efficiency`.  

5. **Model Training**  
   - Algorithms: Logistic Regression, Random Forest, XGBoost, LightGBM.  
   - Ensemble model for better accuracy.  

6. **Model Evaluation**  
   - Metrics: Accuracy, Precision, Recall, F1-score, ROC-AUC.  
   - Feature importance analysis.  

7. **Deployment (Streamlit)**  
   - Predict attrition for new employees.  
   - Display synthetic sentiment.  
   - Provide analytics dashboard (Attrition vs Sentiment, Job Satisfaction trends).  

---

## 🖥️ Streamlit Dashboard  
Run the app locally:  
```bash
streamlit run dashboard/app.py

=======
# employee-attrition-prediction
>>>>>>> 1151005 (first commit)
