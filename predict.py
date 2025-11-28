import joblib
import numpy as np
import pandas as pd

# ✅ Load saved model and label encoder
model = joblib.load("trained_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# ✅ Load the same symptoms list used during training
df = pd.read_csv("dataset/training_data.csv")
symptom_list = df.columns[:-1]  # all columns except 'prognosis'

# ✅ Input symptoms (Change these for testing)
input_symptoms = ["headache", "vomiting", "fatigue"]

# ✅ Create input vector (0s and 1s)
input_vector = [1 if symptom in input_symptoms else 0 for symptom in symptom_list]

# ✅ Predict
prediction = model.predict([input_vector])[0]
predicted_disease = label_encoder.inverse_transform([prediction])[0]

print("🩺 Predicted Disease:", predicted_disease)
