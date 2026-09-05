import pandas as pd

# Load the cleaned data from last step
df = pd.read_csv("data/clean_data.csv")

# Drop rows where LanguageHaveWorkedWith is empty (no skills = nothing to learn from)
df = df.dropna(subset=["LanguageHaveWorkedWith"])

# Split the semicolon-separated skills into a real list
df["skills_list"] = df["LanguageHaveWorkedWith"].apply(lambda x: x.split(";"))

# Let's see an example: before vs after
print("BEFORE splitting (raw text):")
print(df["LanguageHaveWorkedWith"].iloc[0])

print("\nAFTER splitting (real list):")
print(df["skills_list"].iloc[0])

# Now let's count: which skills are most common overall?
all_skills = df["skills_list"].explode()  # flattens every list into individual rows
skill_counts = all_skills.value_counts()

print("\nTop 15 most common skills across everyone:")
print(skill_counts.head(15))
# Turn skills into one-hot encoded columns (1 = knows this skill, 0 = doesn't)
top_skills = skill_counts.head(30).index.tolist()  # only use top 30 skills to keep it manageable

for skill in top_skills:
    df[f"skill_{skill}"] = df["skills_list"].apply(lambda x: 1 if skill in x else 0)

print("\nNew columns added! Here's a preview:")
print(df[[f"skill_{s}" for s in top_skills[:5]]].head())

# Save this enriched version for the next step (model training)
df.to_csv("data/features_ready.csv", index=False)
print("\nSaved features_ready.csv successfully!")