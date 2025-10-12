# Employee Attrition Prediction Project

## Overview
This project aims to predict employee attrition using HR data combined with sentiment analysis derived from employee feedback. It leverages machine learning models to identify key factors driving attrition and provides an interactive dashboard for insights and predictions. Built with Python, Streamlit, and various data science libraries, this project is designed for HR professionals and data analysts to understand and mitigate employee turnover.

## Project Structure
```
📦 Employee Attrition Prediction
    ├── .gitignore              # Git ignore file for version control
    ├── README.md               # Project documentation
    ├── app.py                  # Streamlit application for dashboard and prediction
    ├── data                    # Data storage directory
    │   ├── processed           # Processed datasets
    │   │   ├── employee_attrition_dataset_cleaned.csv
    │   │   └── employee_attrition_with_sentiment.csv
    │   └── raw                 # Raw data archives
    │       └── archive.zip
    ├── models                  # Trained machine learning models and analysis
    │   ├── best_model.pkl      # Saved best-performing model
    │   └── feature_importance.csv # Feature importance scores
    ├── notebooks               # Jupyter notebooks for exploratory analysis
    │   ├── eda.ipynb           # Exploratory Data Analysis
    │   ├── preprocessing.ipynb # Data preprocessing steps
    │   └── sentiment.ipynb     # Sentiment analysis implementation
    ├── requirements.txt        # Python dependencies
    └── src                     # Source code directory
        ├── data_preprocessing.py # Data loading and preprocessing logic
        └── model_training.py    # Model training and evaluation script
```

## Features
- **Data Processing**: Cleans and preprocesses HR data, including sentiment analysis using VADER.
- **Machine Learning**: Trains models (Logistic Regression, Random Forest, XGBoost) with scaling and SMOTE for imbalance handling.
- **Dashboard**: Interactive Streamlit app with visualizations for attrition rate, sentiment distribution, and feature importance.
- **Prediction**: Allows input of employee details to predict attrition probability.

## Installation
1. **Clone the Repository**:
   ```bash
   git clone <https://github.com/sangsthita/employee-attrition-prediction.git>
   cd Employee-Attrition-Prediction
   ```

2. **Set Up Environment**:
   - Ensure Python 3.11+ is installed.
   - Create a virtual environment (optional but recommended):
     ```bash
     python -m venv venv
     source venv/bin/activate  # On Windows: venv\Scripts\activate
     ```

3. **Install Dependencies**:
   Install the required packages using the provided `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare Data**:
   - Ensure the `data/processed/` directory contains `employee_attrition_with_sentiment.csv`.
   - If starting from raw data, unzip `archive.zip` in `data/raw/` and follow preprocessing steps in `notebooks/preprocessing.ipynb`.

5. **Train the Model**:
   Run the training script to generate the model and feature importance files:
   ```bash
   python src/model_training.py
   ```

## Usage
1. **Launch the Dashboard**:
   Run the Streamlit app to interact with the dashboard and prediction tool:
   ```bash
   streamlit run app.py
   ```
   - Access the app at `http://localhost:8501` in your browser.
   - Navigate using the sidebar: **Home**, **Dashboard**, or **Prediction**.

2. **Explore Notebooks**:
   - Open `notebooks/eda.ipynb` for initial data exploration.
   - Use `notebooks/preprocessing.ipynb` for data cleaning and feature engineering.
   - Review `notebooks/sentiment.ipynb` for sentiment analysis implementation.

## Dependencies
The project relies on the following libraries (listed in `requirements.txt`):
- `pandas`
- `numpy`
- `scikit-learn`
- `streamlit`
- `imbalanced-learn`
- `xgboost`
- `matplotlib`
- `seaborn`
- `joblib`
- `vaderSentiment`
- `plotly`

## Contributing
Contributions are welcome! Please fork the repository and submit pull requests with improvements or bug fixes. Ensure to:
- Update the `README.md` with details of changes.
- Add tests if applicable.
- Follow the existing code style.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details (if applicable).

## Contact
For questions or support, please open an issue on the repository or contact the project maintainer.

