# 🎯 Skill ROI Career Recommender

An ML-powered tool that predicts a developer's salary based on their skills, experience, and location — and recommends which skill to learn next for the highest expected salary boost.

## 💡 Problem it solves
Developers often pick up new skills based on generic advice ("learn Python", "learn Cloud") without knowing the actual expected impact. This project uses real survey data from 20,000+ developers to answer: **"Which skill should I learn next for the biggest salary increase?"**

## 🏗️ How it works
1. **Data**: Stack Overflow Developer Survey 2024 (~49,000 responses, filtered to ~20,000 usable records)
2. **Feature Engineering**: Extracted top 30 programming skills, one-hot encoded skills/country/education/company size
3. **Model**: Random Forest Regressor predicting salary (`ConvertedCompYearly`)
   - Mean Absolute Error: ~$29,300
   - R² Score: 0.522
4. **Recommendation Engine**: Simulates adding each candidate skill to a user's profile and measures the predicted salary change, filtered to only skills with 500+ real people as evidence (avoids unreliable predictions from rare skills)
5. **Serving**: FastAPI backend + Streamlit frontend for live, interactive predictions

## 🖥️ Tech Stack
Python, pandas, scikit-learn, FastAPI, Streamlit, joblib

## 🚀 Running it locally
```bash
pip install -r requirements.txt

# Terminal 1: start the API
uvicorn api:app --reload

# Terminal 2: start the interface
streamlit run app.py
```
Then open `http://localhost:8501`

## ⚠️ Known Limitations (honest reflection)
- The model shows strong correlation between rare/specialized skills (e.g. Scala, Perl) and higher salaries. This likely reflects **confounding factors** — such skills tend to be held by senior engineers in high-paying markets/industries — rather than a direct causal effect of learning the skill itself. Recommendations should be read as "skills associated with higher-paying roles," not guaranteed individual outcomes.
- Predictions are based on survey self-reports, which can be noisy or inconsistent.
- Currently only recommends skills with 500+ people as evidence, to reduce noise from rare-skill overfitting.

## 📊 Example Output
| Skill | Predicted New Salary | Salary Boost | Sample Size |
|---|---|---|---|
| Bash/Shell | $16,404 | +$6,212 | 11,104 |
| Kotlin | $12,544 | +$2,352 | 2,428 |
| C | $10,963 | +$771 | 4,095 |

## 🔮 Future Improvements
- Add SHAP explainability to show *why* each recommendation was made
- Deploy live on Render/Railway with a public demo link
- Add drift monitoring to track model performance as job market data changes over time