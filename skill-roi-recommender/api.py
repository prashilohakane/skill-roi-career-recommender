from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI()

model = joblib.load("salary_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")
skill_columns = joblib.load("skill_columns.pkl")
original_df = pd.read_csv("data/features_ready.csv")

skill_sample_sizes = {
    col.replace("skill_", ""): original_df[col].sum() for col in skill_columns
}

class UserProfile(BaseModel):
    skills: list[str]
    work_exp: float
    years_code: float
    country: str

@app.post("/recommend")
def recommend(user: UserProfile):
    profile = {col: 0 for col in feature_columns}

    for skill in user.skills:
        col_name = f"skill_{skill}"
        if col_name in profile:
            profile[col_name] = 1

    profile["WorkExp"] = user.work_exp
    profile["YearsCode"] = user.years_code

    country_col = f"Country_{user.country}"
    if country_col in profile:
        profile[country_col] = 1

    profile_df = pd.DataFrame([profile])[feature_columns]
    current_salary = model.predict(profile_df)[0]

    results = []
    for skill_col in skill_columns:
        skill_name = skill_col.replace("skill_", "")
        if skill_name in user.skills:
            continue
        if skill_sample_sizes[skill_name] < 500:
            continue

        new_profile = profile.copy()
        new_profile[skill_col] = 1
        new_profile_df = pd.DataFrame([new_profile])[feature_columns]
        new_salary = model.predict(new_profile_df)[0]

        results.append({
            "skill": skill_name,
            "predicted_new_salary": round(new_salary, 2),
            "salary_boost": round(new_salary - current_salary, 2),
            "sample_size": int(skill_sample_sizes[skill_name])
        })

    results = sorted(results, key=lambda x: x["salary_boost"], reverse=True)[:10]

    return {
        "predicted_current_salary": round(current_salary, 2),
        "top_recommendations": results
    }