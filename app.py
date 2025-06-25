from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

with open("disease_model.pkl", "rb") as f:
    model, symptoms_list, disease_names = pickle.load(f)

advice_map = {
    "Malaria": "Stay hydrated and see a doctor for antimalarial meds.",
    "Common Cold": "Rest, stay warm, and take fluids.",
    "Typhoid": "Visit a physician. You may need antibiotics.",
    "Diabetes": "Control sugar intake, exercise, and consult an endocrinologist.",
    "Dengue": "Drink water, monitor platelets, consult a hospital.",
    # Add more if needed
}

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    advice = None
    if request.method == "POST":
        selected = request.form.getlist("symptoms")
        input_vector = [1 if s in selected else 0 for s in symptoms_list]
        df = pd.DataFrame([input_vector], columns=symptoms_list)
        idx = model.predict(df)[0]
        prediction = disease_names[idx]
        advice = advice_map.get(prediction, "Please consult a doctor for detailed guidance.")
    return render_template("index.html", symptoms_list=symptoms_list, prediction=prediction, advice=advice)

if __name__ == "__main__":
    app.run(debug=True)
