import pandas as pd
import joblib

# Load our saved model and feature lists
model = joblib.load("salary_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")
skill_columns = joblib.load("skill_columns.pkl")

# ---- STEP 1: Define a person's current profile ----
# Change these values to match YOUR skills for a fun real test!
my_skills = ["Python", "SQL"]       # skills you currently know
my_work_exp = 1                     # years of work experience
my_years_code = 2                   # years of coding overall
my_country = "India"   # change to your actual country
# ---- STEP 2: Build a row of data matching the model's expected format ----
# Start with all zeros for every feature
profile = {col: 0 for col in feature_columns}

# Turn on the skills this person already knows
for skill in my_skills:
    col_name = f"skill_{skill}"
    if col_name in profile:
        profile[col_name] = 1

profile["WorkExp"] = my_work_exp
profile["YearsCode"] = my_years_code
country_col = f"Country_{my_country}"
if country_col in profile:
    profile[country_col] = 1
else:
    print(f"Warning: '{my_country}' not found in training data as a distinct category.")
# ---- STEP 3: Predict current salary ----
profile_df = pd.DataFrame([profile])[feature_columns]
current_salary = model.predict(profile_df)[0]
print(f"Predicted CURRENT salary: ${current_salary:,.2f}\n")

# ---- STEP 4: Try adding each new skill one at a time, see the impact ----
results = []

for skill_col in skill_columns:
    skill_name = skill_col.replace("skill_", "")
    
    if skill_name in my_skills:
        continue  # skip skills they already know
    
    new_profile = profile.copy()
    new_profile[skill_col] = 1  # simulate learning this new skill
    
    new_profile_df = pd.DataFrame([new_profile])[feature_columns]
    new_salary = model.predict(new_profile_df)[0]
    
    salary_boost = new_salary - current_salary
    results.append((skill_name, new_salary, salary_boost))

# ---- STEP 5: Rank skills by biggest salary boost ----
# Load original data to count how many people actually know each skill (for trust/reliability)
original_df = pd.read_csv("data/features_ready.csv")

skill_sample_sizes = {}
for skill_col in skill_columns:
    skill_sample_sizes[skill_col.replace("skill_", "")] = original_df[skill_col].sum()

results_df = pd.DataFrame(results, columns=["Skill", "Predicted_New_Salary", "Salary_Boost"])
results_df["People_Who_Know_This"] = results_df["Skill"].map(skill_sample_sizes)

# Only trust recommendations backed by a reasonable sample size (avoids noisy rare-skill predictions)
reliable_results = results_df[results_df["People_Who_Know_This"] >= 500]
reliable_results = reliable_results.sort_values(by="Salary_Boost", ascending=False)

print("TOP 10 SKILLS TO LEARN NEXT (only showing skills with 500+ people as evidence):\n")
print(reliable_results.head(10).to_string(index=False))
# SANITY CHECK: compare real average salary for people who know Scala vs those who don't
# This helps verify if the model's prediction matches reality, or is just noise

check_skill = "skill_Scala"
known = original_df[original_df[check_skill] == 1]["ConvertedCompYearly"]
unknown = original_df[original_df[check_skill] == 0]["ConvertedCompYearly"]

print(f"\nSANITY CHECK for Scala:")
print(f"Average salary (knows Scala): ${known.mean():,.2f}  (sample size: {len(known)})")
print(f"Average salary (doesn't know Scala): ${unknown.mean():,.2f}  (sample size: {len(unknown)})")