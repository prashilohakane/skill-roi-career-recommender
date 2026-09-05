import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

df = pd.read_csv("data/features_ready.csv")

print("Before filtering:", df.shape)
df = df[(df["ConvertedCompYearly"] >= 5000) & (df["ConvertedCompYearly"] <= 300000)]
print("After filtering:", df.shape)

skill_columns = [col for col in df.columns if col.startswith("skill_")]

df["WorkExp"] = df["WorkExp"].fillna(0)

df["YearsCode"] = pd.to_numeric(df["YearsCode"], errors="coerce")
df["YearsCode"] = df["YearsCode"].fillna(0)

df = pd.get_dummies(df, columns=["Country", "EdLevel", "OrgSize"], dummy_na=True)

country_cols = [col for col in df.columns if col.startswith("Country_")]
ed_cols = [col for col in df.columns if col.startswith("EdLevel_")]
org_cols = [col for col in df.columns if col.startswith("OrgSize_")]

feature_columns = skill_columns + ["WorkExp", "YearsCode"] + country_cols + ed_cols + org_cols

X = df[feature_columns]
y = df["ConvertedCompYearly"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"Mean Absolute Error: ${mae:,.2f}")
print(f"R2 Score: {r2:.3f}")
import joblib

# Save the trained model and the exact feature columns it expects
joblib.dump(model, "salary_model.pkl")
joblib.dump(feature_columns, "feature_columns.pkl")
joblib.dump(skill_columns, "skill_columns.pkl")

print("\nModel and feature list saved successfully!")