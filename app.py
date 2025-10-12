import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
from src.data_preprocessing import load_data, get_features_target, compute_sentiment

# Set page config
st.set_page_config(page_title="Employee Attrition Prediction", layout="wide", initial_sidebar_state="expanded")

# Load data
# data_path = "F:\\Sangsthita-Project\\employee-attrition-prediction\\data\\processed\\employee_attrition_with_sentiment.csv"
data_path = 'data/processed/employee_attrition_with_sentiment.csv'
df = load_data(data_path)

# Mapping
dept_map = {0: 'Sales', 1: 'R&D', 2: 'HR', 3: 'Finance', 4: 'IT'}
job_role_map = {0: 'Sales Executive', 1: 'Research Scientist', 2: 'Lab Technician', 3: 'Manufacturing Director'}
df['department_name'] = df['department'].map(dept_map)
df['job_role_name'] = df['job_role'].map(job_role_map)

# Load model and feature importance
model_path = 'models/best_model.pkl'
fi_path = 'models/feature_importance.csv'
try:
    model = joblib.load(model_path)
    fi_df = pd.read_csv(fi_path)
except FileNotFoundError:
    st.error("Model or feature importance not found. Please run src/model_training.py.")
    st.stop()

# Sidebar navigation and filters
st.sidebar.title('Navigation')
page = st.sidebar.radio('Go to', ['Home', 'Dashboard', 'Prediction'])

st.sidebar.title('Dashboard Filters')
selected_dept = st.sidebar.selectbox('Department', options=['All'] + list(dept_map.values()))
selected_sentiment = st.sidebar.selectbox('Sentiment Label', options=['All'] + sorted(df['sentiment_label'].unique()))

# Filter data
df_filtered = df.copy()
if selected_dept != 'All':
    df_filtered = df_filtered[df_filtered['department_name'] == selected_dept]
if selected_sentiment != 'All':
    df_filtered = df_filtered[df_filtered['sentiment_label'] == selected_sentiment]

# --- ENHANCED DARK THEME CSS ---
st.markdown("""
    <style>
    /* Main App Background */
    .stApp {
        background-color: #0D1117;
        color: #F5F5F5;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #161B22 !important;
        color: #FFFFFF !important;
    }

    /* Titles & Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #58A6FF !important;
        font-family: 'Arial', sans-serif;
    }

    /* Paragraph & Labels */
    p, div, span, label {
        color: #E6EDF3 !important;
        font-family: 'Arial', sans-serif;
    }

    /* Metric Boxes */
    .stMetric {
        background-color: #21262D !important;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.5);
        color: #FFFFFF !important;
    }

    /* Buttons */
    .stButton>button {
        background-color: #238636 !important;
        color: white !important;
        border: none;
        border-radius: 8px;
        padding: 0.6em 1em;
        font-size: 1em;
        transition: background-color 0.3s;
    }
    .stButton>button:hover {
        background-color: #2EA043 !important;
    }

    /* Inputs */
    .stSelectbox, .stTextInput, .stNumberInput, .stTextArea {
        background-color: #21262D !important;
        color: #F0F6FC !important;
        border: 1px solid #30363D !important;
        border-radius: 6px !important;
    }
    .stSelectbox div, .stTextInput div, .stNumberInput div, .stTextArea div {
        color: #F0F6FC !important;
    }

    /* Plotly Chart Background */
    .js-plotly-plot .plotly .modebar {
        background-color: #161B22 !important;
    }
    .js-plotly-plot .plotly {
        background-color: #0D1117 !important;
    }

    /* Tabs */
    .stTabs [data-testid="stTab"] {
        background-color: #21262D !important;
        color: #FFFFFF !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: #30363D !important;
        color: #58A6FF !important;
    }

    /* Expanders */
    .stExpander {
        background-color: #21262D !important;
        border: 1px solid #30363D !important;
        border-radius: 6px;
    }

    /* Horizontal Line */
    hr {
        border-color: #30363D !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- Pages ---
if page == 'Home':
    st.title('Employee Attrition Prediction')
    st.markdown("""
    ### Overview
    This app predicts employee attrition using HR data and sentiment analysis from feedback. Key features:
    - **Data**: 10,000 records with metrics like age, income, and sentiment scores.
    - **Models**: Logistic Regression, Random Forest, XGBoost with scaling and SMOTE.
    - **Dashboard**: Key insights on attrition and sentiment trends.
    - **Prediction**: Input employee details to assess attrition risk.
    """)

    # Add an expander for additional info
    with st.expander("Learn More"):
        st.markdown("""
        - **Sentiment Analysis**: Uses VADER to score feedback from 1 (negative) to 4.25 (positive).
        - **Model Enhancements**: Features scaling, imbalance handling with SMOTE, and hyperparameter tuning.
        - **Filters**: Interactive filters for department and sentiment in the dashboard.
        """)

elif page == 'Dashboard':
    st.title('Attrition Insights Dashboard')
    
    # Use tabs for organized view
    tab1, tab2 = st.tabs(["Key Metrics & Sentiment", "Model Insights"])
    
    with tab1:
        st.markdown("### Key Metrics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric('Attrition Rate', f'{df_filtered["attrition"].mean() * 100:.2f}%', delta_color="inverse")
        with col2:
            if 'sentiment_score' in df_filtered.columns:
                st.metric('Average Sentiment Score', f'{df_filtered["sentiment_score"].mean():.2f}', delta_color="normal")
        with col3:
            st.metric('Total Employees', len(df_filtered))
        
        st.markdown("<hr>", unsafe_allow_html=True)
        
        st.subheader('Sentiment Label Distribution')
        fig = px.histogram(df_filtered, x='sentiment_label', color='sentiment_label',
                           color_discrete_map={'Negative': '#EF4444', 'Neutral': '#A1A1AA', 'Positive': '#10B981'},
                           labels={'sentiment_label': 'Sentiment'})
        fig.update_layout(
            height=450,
            margin=dict(l=20, r=20, t=50, b=20),
            font=dict(family="Arial", size=12, color="#E6EDF3"),
            plot_bgcolor="#0D1117",
            paper_bgcolor="#0D1117",
            title_font=dict(color="#58A6FF"),
            bargap=0.2,
            showlegend=False
        )
        fig.update_traces(marker_line_width=1, marker_line_color="#30363D")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader('Top 10 Features Driving Attrition')
        fig = px.bar(fi_df.head(10), x='importance', y='feature', orientation='h',
                     color='importance', color_continuous_scale=['#60A5FA', '#1D4ED8'])
        fig.update_layout(
            height=450,
            margin=dict(l=20, r=20, t=50, b=20),
            font=dict(family="Arial", size=12, color="#E6EDF3"),
            plot_bgcolor="#0D1117",
            paper_bgcolor="#0D1117",
            title_font=dict(color="#58A6FF"),
            coloraxis_showscale=False
        )
        fig.update_traces(marker_line_width=1, marker_line_color="#30363D")
        st.plotly_chart(fig, use_container_width=True)

elif page == 'Prediction':
    st.title('Predict Employee Attrition')
    with st.form('prediction_form'):
        st.subheader('Enter Employee Details')
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input('Age', min_value=18, max_value=60, value=30)
            gender_options = [('Female', 0), ('Male', 1)]
            gender = st.selectbox('Gender', options=gender_options, format_func=lambda x: x[0])
            gender = gender[1]
            marital_options = [('Single', 0), ('Married', 1), ('Divorced', 2)]
            marital_status = st.selectbox('Marital Status', options=marital_options, format_func=lambda x: x[0])
            marital_status = marital_status[1]
            dept_options = [('Sales', 0), ('R&D', 1), ('HR', 2), ('Finance', 3), ('IT', 4)]
            department = st.selectbox('Department', options=dept_options, format_func=lambda x: x[0])
            department = department[1]
            role_options = [('Sales Executive', 0), ('Research Scientist', 1), ('Lab Technician', 2), ('Manufacturing Director', 3)]
            job_role = st.selectbox('Job Role', options=role_options, format_func=lambda x: x[0])
            job_role = job_role[1]
            job_level = st.number_input('Job Level', min_value=1, max_value=5, value=3)
        with col2:
            monthly_income = st.number_input('Monthly Income', min_value=1000, max_value=20000, value=5000)
            hourly_rate = st.number_input('Hourly Rate', min_value=30, max_value=100, value=50)
            years_at_company = st.number_input('Years at Company', min_value=0, max_value=40, value=5)
            years_in_current_role = st.number_input('Years in Current Role', min_value=0, max_value=20, value=3)
            years_since_last_promotion = st.number_input('Years Since Last Promotion', min_value=0, max_value=15, value=1)
            work_life_balance = st.number_input('Work Life Balance (1-4)', min_value=1, max_value=4, value=3)
            job_satisfaction = st.number_input('Job Satisfaction (1-4)', min_value=1, max_value=4, value=3)
        with col3:
            performance_rating = st.number_input('Performance Rating (1-4)', min_value=1, max_value=4, value=3)
            training_hours_last_year = st.number_input('Training Hours Last Year', min_value=0, max_value=100, value=50)
            overtime = st.selectbox('Overtime (0=No, 1=Yes)', [0, 1])
            project_count = st.number_input('Project Count', min_value=0, max_value=10, value=3)
            average_hours_worked_per_week = st.number_input('Avg Hours Worked/Week', min_value=30, max_value=60, value=40)
            absenteeism = st.number_input('Absenteeism (days)', min_value=0, max_value=20, value=2)
            work_environment_satisfaction = st.number_input('Work Env Satisfaction (1-4)', min_value=1, max_value=4, value=3)
        with col1:
            relationship_with_manager = st.number_input('Relationship with Manager (1-4)', min_value=1, max_value=4, value=3)
        with col2:
            job_involvement = st.number_input('Job Involvement (1-4)', min_value=1, max_value=4, value=3)
        with col3:
            distance_from_home = st.number_input('Distance from Home (km)', min_value=0, max_value=50, value=5)
        with col1:
            number_of_companies_worked = st.number_input('Number of Companies Worked', min_value=0, max_value=10, value=2)
        feedback_text = st.text_area('Feedback Text', value='I am happy with my job and work-life balance.')
        submit_button = st.form_submit_button(label='Predict Attrition')

    if submit_button:
        input_data = {
            'age': age, 'gender': gender, 'marital_status': marital_status, 'department': department,
            'job_role': job_role, 'job_level': job_level, 'monthly_income': monthly_income,
            'hourly_rate': hourly_rate, 'years_at_company': years_at_company,
            'years_in_current_role': years_in_current_role, 'years_since_last_promotion': years_since_last_promotion,
            'work_life_balance': work_life_balance, 'job_satisfaction': job_satisfaction,
            'performance_rating': performance_rating, 'training_hours_last_year': training_hours_last_year,
            'overtime': overtime, 'project_count': project_count,
            'average_hours_worked_per_week': average_hours_worked_per_week, 'absenteeism': absenteeism,
            'work_environment_satisfaction': work_environment_satisfaction,
            'relationship_with_manager': relationship_with_manager, 'job_involvement': job_involvement,
            'distance_from_home': distance_from_home, 'number_of_companies_worked': number_of_companies_worked,
            'feedback_text': feedback_text
        }
        df_input = pd.DataFrame([input_data])
        X_input, _ = get_features_target(df_input, apply_sentiment=True)
        # Align columns with training data
        missing_cols = set(model.named_steps['scaler'].feature_names_in_) - set(X_input.columns)
        for col in missing_cols:
            X_input[col] = 0
        X_input = X_input[model.named_steps['scaler'].feature_names_in_]
        prediction = model.predict(X_input)[0]
        probability = model.predict_proba(X_input)[0][1]
        st.success(f'**Predicted Attrition**: {"Yes" if prediction == 1 else "No"}')
        st.info(f'**Probability of Attrition**: {probability:.2f}')