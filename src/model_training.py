import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import pandas as pd
from data_preprocessing import load_data, get_features_target

def train():
    data_path = "F:\\Sangsthita-Project\\employee-attrition-prediction\\data\\processed\\employee_attrition_with_sentiment.csv"
    df = load_data(data_path)
    X, y = get_features_target(df, apply_sentiment=False)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'XGBoost': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    }
    
    best_model = None
    best_score = 0
    best_name = ''
    best_pipeline = None
    
    os.makedirs('../models', exist_ok=True)
    
    for name, model in models.items():
        pipeline = Pipeline([('scaler', StandardScaler()), ('classifier', model)])
        pipeline.fit(X_train_res, y_train_res)
        pred = pipeline.predict(X_test)
        score = f1_score(y_test, pred, average='binary')
        print(f"{name} F1-Score: {score}")
        print(classification_report(y_test, pred))
        if score > best_score:
            best_score = score
            best_pipeline = pipeline
            best_name = name
    
    # Hyperparameter tuning for the best model
    if best_name == 'Logistic Regression':
        param_grid = {'classifier__C': [0.1, 1, 10]}
    elif best_name == 'Random Forest':
        param_grid = {'classifier__n_estimators': [100, 200], 'classifier__max_depth': [None, 10, 20]}
    elif best_name == 'XGBoost':
        param_grid = {'classifier__n_estimators': [100, 200], 'classifier__learning_rate': [0.01, 0.1]}
    
    gs = GridSearchCV(best_pipeline, param_grid, cv=5, scoring='f1', n_jobs=-1)
    gs.fit(X_train_res, y_train_res)
    best_model = gs.best_estimator_
    print(f"Best tuned model ({best_name}) parameters: {gs.best_params_}")
    
    # Save the best model
    joblib.dump(best_model, '../models/best_model.pkl')
    print(f"Best model ({best_name}) saved with F1-score: {best_score}")
    
    # Feature importance
    classifier = best_model.named_steps['classifier']
    if best_name == 'Logistic Regression':
        importance = classifier.coef_[0]
    else:
        importance = classifier.feature_importances_
    fi_df = pd.DataFrame({'feature': X.columns, 'importance': importance}).sort_values('importance', ascending=False)
    fi_df.to_csv('../models/feature_importance.csv', index=False)
    print("Feature importance saved to models/feature_importance.csv")

if __name__ == '__main__':
    train()