import streamlit as st
import requests

st.title("🎯 Skill ROI Career Recommender")
st.write("Find out which skill to learn next for the biggest salary boost.")

# Input fields
skills_input = st.text_input("Enter your current skills (comma-separated)", "Python, SQL")
work_exp = st.number_input("Years of work experience", min_value=0, value=1)
years_code = st.number_input("Years of coding overall", min_value=0, value=2)
country = st.text_input("Your country", "India")

if st.button("Get Recommendations"):
    skills_list = [s.strip() for s in skills_input.split(",")]

    payload = {
        "skills": skills_list,
        "work_exp": work_exp,
        "years_code": years_code,
        "country": country
    }

    response = requests.post("http://127.0.0.1:8000/recommend", json=payload)

    if response.status_code == 200:
        data = response.json()
        st.subheader(f"Predicted Current Salary: ${data['predicted_current_salary']:,.2f}")

        st.subheader("Top Skills to Learn Next:")
        for rec in data["top_recommendations"]:
            st.write(f"**{rec['skill']}** → Predicted salary: ${rec['predicted_new_salary']:,.2f} "
                     f"(boost: +${rec['salary_boost']:,.2f}, based on {rec['sample_size']} people)")
    else:
        st.error("Something went wrong. Make sure the API server is running.")