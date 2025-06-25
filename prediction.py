# prediction.py

import pickle
import pandas as pd

# Load model and metadata
with open("disease_model.pkl", "rb") as f:
    model, symptoms_list, class_names = pickle.load(f)

# Sample user-selected symptoms
user_symptoms = [
    "fatigue", "cough", "high_fever"
]

# Build input vector from symptoms
input_vector = [1 if symptom in user_symptoms else 0 for symptom in symptoms_list]
input_df = pd.DataFrame([input_vector], columns=symptoms_list)

# Predict
prediction = model.predict(input_df)[0]
disease_name = class_names[prediction]

print("🩺 Predicted Disease:", disease_name)
