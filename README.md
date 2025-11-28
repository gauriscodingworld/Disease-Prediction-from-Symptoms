# Disease-Prediction-from-Symptoms
A machine learning–powered web app that predicts the most likely disease based on user-selected symptoms. Built using Random Forest Classifier, Gradio UI, and Bing Search API for real-time remedy suggestions.

🚀 Features

✔️ Random Forest Classifier trained on a labeled symptom–disease dataset

✔️ Symptom Input UI using Gradio

✔️ Binary Vector Encoding for symptom representation

✔️ Fast & accurate prediction of the probable disease

✔️ Live treatment/remedy search via Bing Search API

✔️ Simple, interactive, and beginner-friendly workflow

🧠 How It Works

Training

A dataset of symptoms + diagnosis (prognosis) is used

Labels are encoded using LabelEncoder

Model trained with RandomForestClassifier

User Input

User selects symptoms in the Gradio interface

Vectorization

Symptoms → Binary vector (1 = selected, 0 = not selected)

Prediction

Vector passed into the model

Output: predicted disease

Remedy Search

Bing Search API fetches treatment info

Articles/results shown to the user

📦 Tech Stack

Python

Scikit-learn – ML model training

Gradio – Web interface

Pandas / NumPy – Data processing

Bing Search API – Treatment/solution retrieval

🗂️ Files

model.pkl – Trained Random Forest model

app.py – Main application

dataset.csv – Symptoms + diseases

vectorizer.py – Binary encoding logic

▶️ Running the Project
pip install -r requirements.txt
python app.py


Gradio link will appear—open it in the browser.

📚 Credits / References

Scikit-learn Docs

Kaggle (Dataset Source)

Bing Search API Docs
