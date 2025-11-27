import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# Load dataset
df = pd.read_csv("Dataset.csv")

# Combine features into text
df["combined"] = (
    df["Cuisines"].astype(str) + " "
    + df["City"].astype(str) + " "
    + df["Price range"].astype(str)
)

# Vectorize
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(df["combined"])

# Save
joblib.dump(tfidf, "vectorizer.pkl")
joblib.dump(tfidf_matrix, "tfidf_matrix.pkl")

print("Training complete — vectorizer.pkl & tfidf_matrix.pkl created.")
