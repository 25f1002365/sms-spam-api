"""Train the SMS spam classifier and export it to model.pkl."""
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# 1. Load data (label \t message)
df = pd.read_csv("data/sms.tsv", sep="\t", header=None, names=["label", "message"])
df["target"] = (df["label"] == "spam").astype(int)

# 2. Split
X_train, X_test, y_train, y_test = train_test_split(
    df["message"], df["target"], test_size=0.2, random_state=42, stratify=df["target"]
)

# 3. One pipeline = text cleaning + model, saved together as a single file
model = Pipeline([
    ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2))),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
])
model.fit(X_train, y_train)

# 4. Evaluate
preds = model.predict(X_test)
print("Accuracy:", round(accuracy_score(y_test, preds), 4))
print(classification_report(y_test, preds, target_names=["ham", "spam"]))

# 5. Export
joblib.dump(model, "model.pkl")
print("Saved model.pkl")
