import pandas as pd

df = pd.read_csv("data/survey_results_public.csv", low_memory=False)

# Pick only the columns we actually need
columns_needed = [
    "LanguageHaveWorkedWith",
    "DatabaseHaveWorkedWith",
    "PlatformHaveWorkedWith",
    "WebframeHaveWorkedWith",
    "YearsCode",
    "WorkExp",
    "DevType",
    "EdLevel",
    "Country",
    "OrgSize",
    "ConvertedCompYearly",
    "JobSat"
]

df_clean = df[columns_needed]

# Drop rows where salary is missing (we can't train on those)
df_clean = df_clean.dropna(subset=["ConvertedCompYearly"])

print("Shape after cleaning:", df_clean.shape)
print(df_clean.head())

# Save the cleaned version so we don't have to redo this every time
df_clean.to_csv("data/clean_data.csv", index=False)
print("Saved clean_data.csv successfully!")