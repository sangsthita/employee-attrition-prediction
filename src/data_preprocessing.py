import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def load_data(file_path="F:\\Sangsthita-Project\\employee-attrition-prediction\\data\\processed\\employee_attrition_with_sentiment.csv"):
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file not found at {file_path}. Please check the path.")

def compute_sentiment(df):
    if 'feedback_text' not in df.columns:
        raise ValueError("Feedback text column is required for sentiment computation.")
    analyzer = SentimentIntensityAnalyzer()
    # Scale VADER compound (-1 to 1) to approx 1-4.25 to match dataset range
    df['sentiment_score'] = df['feedback_text'].apply(
        lambda x: ((analyzer.polarity_scores(x)['compound'] + 1) / 2) * 3.25 + 1
    )
    return df

def preprocess_data(df, apply_sentiment=False):
    if apply_sentiment:
        df = compute_sentiment(df)
    # Drop unnecessary columns
    df = df.drop(['employee_id', 'sentiment_label', 'feedback_text'], axis=1, errors='ignore')
    # One-hot encode categorical columns
    cat_cols = ['marital_status', 'department', 'job_role']
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    return df

def get_features_target(df, apply_sentiment=False):
    df_processed = preprocess_data(df, apply_sentiment)
    X = df_processed.drop('attrition', axis=1, errors='ignore')
    y = df_processed['attrition'] if 'attrition' in df_processed.columns else None
    return X, y