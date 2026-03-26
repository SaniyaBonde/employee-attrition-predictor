from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import uvicorn

app = FastAPI(title="Employee Attrition Predictor")

saved = joblib.load("best_model.pkl")
model_pipeline = saved['model']
threshold = saved['threshold']

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

class EmployeeData(BaseModel):
    Age: int
    Department: str
    DistanceFromHome: int
    Education: int
    EnvironmentSatisfaction: int
    JobRole: str
    JobSatisfaction: int
    MaritalStatus: str
    MonthlyIncome: int
    NumCompaniesWorked: int
    OverTime: str
    TotalWorkingYears: int
    WorkLifeBalance: int
    YearsAtCompany: int

@app.get("/")
def root():
    return {"message": "Attrition Predictor API is running"}

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model_pipeline is not None}

@app.post("/predict")
def predict(data: EmployeeData):
    try:
        input_data = pd.DataFrame([{
            # User provided fields
            'Age':                      data.Age,
            'Department':               data.Department,
            'DistanceFromHome':         data.DistanceFromHome,
            'Education':                data.Education,
            'EnvironmentSatisfaction':  data.EnvironmentSatisfaction,
            'JobRole':                  data.JobRole,
            'JobSatisfaction':          data.JobSatisfaction,
            'MaritalStatus':            data.MaritalStatus,
            'MonthlyIncome':            data.MonthlyIncome,
            'NumCompaniesWorked':       data.NumCompaniesWorked,
            'OverTime':                 data.OverTime,
            'TotalWorkingYears':        data.TotalWorkingYears,
            'WorkLifeBalance':          data.WorkLifeBalance,
            'YearsAtCompany':           data.YearsAtCompany,

            # Defaults for hidden fields
            'BusinessTravel':           'Travel_Rarely',
            'DailyRate':                800,
            'EducationField':           'Life Sciences',
            'Gender':                   'Male',
            'HourlyRate':               60,
            'JobInvolvement':           3,
            'JobLevel':                 2,
            'MonthlyRate':              15000,
            'PercentSalaryHike':        15,
            'PerformanceRating':        3,
            'RelationshipSatisfaction': 3,
            'StockOptionLevel':         1,
            'TrainingTimesLastYear':    2,
            'YearsInCurrentRole':       max(data.YearsAtCompany // 2, 1),
            'YearsSinceLastPromotion':  1,
            'YearsWithCurrManager':     min(3, data.YearsAtCompany),
        }])[ALL_FEATURES]

        probability = model_pipeline.predict_proba(input_data)[0][1]
        prediction = 1 if probability >= threshold else 0

        return {
            'attrition_risk':  'High' if prediction == 1 else 'Low',
            'probability':     round(float(probability), 3),
            'risk_level':      ('High'   if probability >= 0.6 else
                                'Medium' if probability >= 0.35 else 'Low')
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)