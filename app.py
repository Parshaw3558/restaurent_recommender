import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------- LOAD MODELS ----------------------
df = pd.read_csv("Dataset.csv")
cv = joblib.load("vectorizer.pkl")
tfidf_matrix = joblib.load("tfidf_matrix.pkl")

st.title("🍽 Restaurant Recommendation System")
st.write("Enter your preferences to get restaurant recommendations.")

# ---------------------- FUNCTION (USE IT HERE) ----------------------
def recommend_restaurant(pref):
    vec = cv.transform([pref])                          # Convert user input
    scores = cosine_similarity(vec, tfidf_matrix).flatten()   # Similarity
    idx = scores.argsort()[-5:][::-1]                  # Top 5
    result = df.iloc[idx][[
        "Restaurant Name", 
        "Cuisines", 
        "City", 
        "Price range", 
        "Aggregate rating"
    ]]
    return result

# ---------------------- USER INPUT ----------------------
cuisine = st.text_input("Cuisine (example: Chinese, Italian, North Indian)")
city = st.text_input("City (example: Pune, Mumbai, Bangalore)")
price = st.text_input("Price Range (1–4)")

# ---------------------- BUTTON ----------------------
if st.button("Get Recommendations"):
    pref = f"{cuisine} {city}".strip()
    if price:
        pref += f" price{price}"

    if pref.strip() == "":
        st.warning("Please enter at least one input.")
    else:
        result = recommend_restaurant(pref)
        st.success("Here are your top restaurant matches:")
        st.dataframe(result)
