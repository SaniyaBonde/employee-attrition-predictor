import streamlit as st
import joblib
import pandas as pd

# --- Load model ---
saved = joblib.load('best_model.pkl')
model_pipeline = saved['model']
threshold = saved['threshold']

# --- Exact columns model was trained on ---
NUMERICAL = [
    'Age', 'DailyRate', 'DistanceFromHome', 'Education',
    'EnvironmentSatisfaction', 'HourlyRate', 'JobInvolvement', 'JobLevel',
    'JobSatisfaction', 'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked',
    'PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction',
    'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear',
    'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole',
    'YearsSinceLastPromotion', 'YearsWithCurrManager'
]

CATEGORICAL = [
    'BusinessTravel', 'Department', 'EducationField',
    'Gender', 'JobRole', 'MaritalStatus', 'OverTime'
]

ALL_FEATURES = NUMERICAL + CATEGORICAL

# --- Page config ---
st.set_page_config(page_title="Employee Attrition Predictor", layout="centered")
st.title("Employee Attrition Predictor")
st.write("Fill in employee details below to predict attrition risk.")

# --- Input form ---
with st.form("employee_form"):
    st.subheader("Employee details")

    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", 18, 60, 30)
        monthly_income = st.number_input("Monthly Income ($)", 1000, 20000, 5000, step=500)
        job_satisfaction = st.selectbox(
            "Job Satisfaction",
            [1, 2, 3, 4],
            format_func=lambda x: {1:"Low", 2:"Medium", 3:"High", 4:"Very High"}[x]
        )
        overtime = st.selectbox("Overtime", ["No", "Yes"])
        department = st.selectbox(
            "Department",
            ["Sales", "Research & Development", "Human Resources"]
        )
        job_role = st.selectbox(
            "Job Role",
            ["Sales Executive", "Research Scientist", "Lab Technician",
             "Manufacturing Director", "Healthcare Representative",
             "Manager", "Sales Representative", "Research Director",
             "Human Resources"]
        )

    with col2:
        distance = st.slider("Distance From Home (km)", 1, 29, 5)
        years_at_company = st.slider("Years at Company", 0, 40, 3)
        work_life = st.selectbox(
            "Work-Life Balance",
            [1, 2, 3, 4],
            format_func=lambda x: {1:"Bad", 2:"Good", 3:"Better", 4:"Best"}[x]
        )
        marital_status = st.selectbox(
            "Marital Status",
            ["Single", "Married", "Divorced"]
        )
        business_travel = st.selectbox(
            "Business Travel",
            ["Non-Travel", "Travel_Rarely", "Travel_Frequently"]
        )
        environment_satisfaction = st.selectbox(
            "Environment Satisfaction",
            [1, 2, 3, 4],
            format_func=lambda x: {1:"Low", 2:"Medium", 3:"High", 4:"Very High"}[x]
        )

    submitted = st.form_submit_button("Predict Attrition Risk", type="primary")

# --- Prediction ---
if submitted:
    input_data = pd.DataFrame([{
        # Numerical — user inputs
        'Age':                      age,
        'MonthlyIncome':            monthly_income,
        'JobSatisfaction':          job_satisfaction,
        'DistanceFromHome':         distance,
        'YearsAtCompany':           years_at_company,
        'WorkLifeBalance':          work_life,
        'EnvironmentSatisfaction':  environment_satisfaction,

        # Numerical — sensible defaults
        'DailyRate':                800,
        'Education':                3,
        'HourlyRate':               60,
        'JobInvolvement':           3,
        'JobLevel':                 2,
        'MonthlyRate':              15000,
        'NumCompaniesWorked':       2,
        'PercentSalaryHike':        15,
        'PerformanceRating':        3,
        'RelationshipSatisfaction': 3,
        'StockOptionLevel':         1,
        'TotalWorkingYears':        max(years_at_company, 1) + 2,
        'TrainingTimesLastYear':    2,
        'YearsInCurrentRole':       max(years_at_company // 2, 1),
        'YearsSinceLastPromotion':  1,
        'YearsWithCurrManager':     min(3, years_at_company),

        # Categorical — user inputs
        'Department':               department,
        'JobRole':                  job_role,
        'MaritalStatus':            marital_status,
        'BusinessTravel':           business_travel,
        'OverTime':                 overtime,

        # Categorical — sensible defaults
        'EducationField':           'Life Sciences',
        'Gender':                   'Male',
    }])

    # Force exact column order matching training
    input_data = input_data[ALL_FEATURES]

    # Predict
    prob = model_pipeline.predict_proba(input_data)[0][1]
    prediction = 1 if prob >= threshold else 0

    # Display result
    st.divider()

    if prediction == 1:
        st.error(f"### High Attrition Risk")
    else:
        st.success(f"### Low Attrition Risk")

    col1, col2 = st.columns(2)
    col1.metric("Attrition Probability", f"{prob * 100:.1f}%")
    col2.metric("Threshold Used", f"{threshold * 100:.0f}%")

    # Probability bar
    st.write("**Risk level**")
    st.progress(float(prob))

    # What drove this prediction
    st.divider()
    st.subheader("Key inputs used")
    summary = pd.DataFrame({
        "Feature": [
            "Age", "Monthly Income", "Overtime", "Distance From Home",
            "Years at Company", "Job Satisfaction", "Work-Life Balance",
            "Environment Satisfaction", "Department", "Marital Status",
            "Business Travel"
        ],
        "Value": [
            age, f"${monthly_income:,}", overtime, distance,
            years_at_company, job_satisfaction, work_life,
            environment_satisfaction, department, marital_status,
            business_travel
        ]
    })
    st.dataframe(summary, use_container_width=True, hide_index=True)