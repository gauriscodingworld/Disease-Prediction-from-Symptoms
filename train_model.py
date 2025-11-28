import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# Load dataset
df = pd.read_csv("dataset/training_data.csv") # Change path if needed

# Separate features (symptoms) and target (disease)
X = df.drop(columns=["prognosis"])  # ✅ only symptoms
y = df["prognosis"]                 # ✅ only target

# Encode the target labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train the model
model = RandomForestClassifier()
model.fit(X, y_encoded)

# Make sure saved_model directory exists
os.makedirs("saved_model", exist_ok=True)

# Save model and encoder
joblib.dump(model, "saved_model/random_forest.joblib")
joblib.dump(le, "saved_model/label_encoder.joblib")

# Save symptom list (column names of X)
joblib.dump(X.columns.tolist(), "saved_model/symptoms_list.pkl")

print("✅ Model trained and saved successfully")


