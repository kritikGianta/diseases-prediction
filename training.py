# training.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load the dataset
df = pd.read_csv("Training.csv")

# Split features and target
X = df.drop(columns=["prognosis"])
y_raw = df["prognosis"]

# Encode target labels
y_categorical = y_raw.astype("category")
y = y_categorical.cat.codes
class_names = list(y_categorical.cat.categories)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model, symptom list, and class names
with open("disease_model.pkl", "wb") as f:
    pickle.dump((model, X.columns.tolist(), class_names), f)

print("✅ Model trained and saved as 'disease_model.pkl'")
