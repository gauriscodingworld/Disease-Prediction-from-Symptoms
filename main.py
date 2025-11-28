import gradio as gr
import joblib
import pandas as pd
from serpapi import GoogleSearch

# ------------------------------
# MODEL & DATA LOADING
# ------------------------------
model = joblib.load("saved_model/random_forest.joblib")
label_encoder = joblib.load("saved_model/label_encoder.joblib")
all_symptoms = joblib.load("saved_model/symptoms_list.pkl")

# ------------------------------
# SERPAPI ARTICLE SEARCH (no site restriction)
# ------------------------------
SERPAPI_KEY = "844969b7df0d988781a979a92a902a0959ab5af99002f7053094d53cc6813f98"  # <<<--- Put your SerpAPI key here

def serpapi_article_search(disease_name):
    params = {
        "engine": "google",
        "q": f"{disease_name} remedy OR treatment",
        "api_key": SERPAPI_KEY,
        "num": 3,
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    articles = []
    for r in results.get('organic_results', []):
        articles.append({
            "title": r.get("title"),
            "snippet": r.get("snippet"),
            "url": r.get("link"),
        })
    return articles

# ------------------------------
# DISEASE INFO (Add more as needed)
# ------------------------------
disease_info = {
    "Fungal infection": {
        "description": "A skin or systemic infection caused by fungi.",
        "causes": "Fungal organisms (dermatophytes, yeasts, molds).",
        "symptoms": ["itching", "skin rash", "nodal skin eruptions"],
        "treatments": "Antifungal creams or oral medications.",
        "resources": [
            {"name": "CDC", "url": "https://www.cdc.gov/fungal/"},
        ],
        "severity": "Consult a doctor if symptoms persist.",
    },
    "Allergy": {
        "description": "An immune response to non-harmful foreign substances.",
        "causes": "Pollen, dust, food, insect stings, medications, etc.",
        "symptoms": ["sneezing", "itching", "rash", "swelling"],
        "treatments": "Antihistamines, avoiding allergens.",
        "resources": [
            {"name": "Mayo Clinic", "url": "https://www.mayoclinic.org/diseases-conditions/allergies/symptoms-causes/syc-20351497"},
        ],
        "severity": "Consult a doctor if severe or persistent.",
    }
    # ... add more diseases with similar structure ...
}

# ------------------------------
# MAIN PREDICT FUNCTION
# ------------------------------
def predict_disease(symptoms_selected):
    if not symptoms_selected:
        return """<div style='color:#ed4d44;font-size:1.1em;'>⚠️ Please select at least one symptom.</div>"""
    
    input_vector = [1 if symptom in symptoms_selected else 0 for symptom in all_symptoms]
    input_df = pd.DataFrame([input_vector], columns=all_symptoms)
    proba = model.predict_proba(input_df)[0]
    prediction_encoded = proba.argmax()
    confidence = proba[prediction_encoded]
    prediction_label = label_encoder.inverse_transform([prediction_encoded])[0]

    info = disease_info.get(prediction_label, {
        "description": "No information available.",
        "causes": "-",
        "symptoms": [],
        "treatments": "-",
        "resources": [],
        "severity": "Consult a doctor.",
    })

    try:
        articles = serpapi_article_search(prediction_label)
    except Exception as e:
        print("SerpAPI error:", e)
        articles = []

    articles_md = ""
    for a in articles[:3]:
        articles_md += (
            f"<div class='article-card'><a href='{a['url']}' target='_blank' class='article-link'>{a['title']}</a>"
            f"<br><span class='article-snippet'>{a['snippet']}</span></div>"
        )
    if not articles_md:
        articles_md = "<div style='color:#899'>No live articles found for this disease.</div>"

    resources_md = ""
    for r in info.get("resources", []):
        resources_md += f"<a href='{r['url']}' target='_blank' class='resource-link'>{r['name']}</a> "

    # Larger, more open result card
    result_html = f"""
    <div class='result-card'>
    <div class='prediction-header'>
        <span class='predicted-disease'>{prediction_label}</span>
        <span class='confidence'>{confidence:.1%} confident</span>
    </div>
    {f"<div class='info-section'><b>Description:</b> {info['description']}</div>" if info.get('description') and info['description'] != '-' else ''}
    {f"<div class='info-section'><b>Causes:</b> {info['causes']}</div>" if info.get('causes') and info['causes'] != '-' else ''}
    {f"<div class='info-section'><b>Symptoms:</b> {', '.join(info['symptoms'])}</div>" if info.get('symptoms') else ''}
    {f"<div class='info-section'><b>Treatments:</b> {info['treatments']}</div>" if info.get('treatments') and info['treatments'] != '-' else ''}
    {f"<div class='info-section'><b>Resources:</b> {resources_md}</div>" if resources_md else ''}
    {f"<div class='info-section'><b>Severity:</b> {info['severity']}</div>" if info.get('severity') and info['severity'] != '-' else ''}
    <hr>
    <div class='articles-header'>📰 <b>Latest Remedies & Articles:</b></div>
    {articles_md}
    </div>
    """
    return result_html

# ------------------------------
# FEEDBACK HANDLER
# ------------------------------
def handle_feedback(text):
    if text and text.strip():
        with open("user_feedback.txt", "a", encoding="utf-8") as f:
            f.write(text.strip()+"\n---\n")
        return "✅ Thank you for your feedback!"
    return "⚠️ Please enter feedback before submitting."

# ------------------------------
# CUSTOM CSS (Streamlit/modern look, now with BIGGER boxes)
# ------------------------------
CUSTOM_CSS = """
body { background: #23272f; }
.gradio-container { width: 100% !important; max-width: 100vw !important; margin: 0 auto; padding: 0; }
.result-card {
  width: 98vw; max-width: 1100px; min-width: 300px;
  margin: 2.5em auto 1.2em auto;
  background: #282c34; border-radius: 22px; box-shadow: 0 6px 38px #10111522;
  padding: 2.2em 4vw 1.8em 4vw;
  color: #ececec; font-size: 1.24em;
  box-sizing: border-box;
}
.prediction-header {
  font-size: 2.3em; font-weight: 800; color: #59b2fa;
  letter-spacing: -.5px; margin-bottom: 0.3em;
  display: flex; flex-wrap: wrap; align-items: center; gap: 30px;
}
.predicted-disease { font-weight: bold; }
.confidence {
  font-size: 1.05em; font-weight: 700; border-radius: 9px;
  color: #88FCBF; background: #23383e; padding: .3em 1.2em;
  letter-spacing: .03em; margin-left: .8em;
}
.info-section { margin-bottom: 0.44em; }
.articles-header { font-size:1.16em; margin-top:.8em; color:#76cbc3;font-weight:600;}
.article-card {
  margin: 0.35em 0 1.25em 0; background:#20242a; border-radius: 10px; padding:.9em 1.1em;
}
.article-link {
  color: #4EC3FA; text-decoration: none; font-weight: 600; font-size:1.17em;
}
.article-link:hover { text-decoration:underline; }
.article-snippet { color:#b1bbc4; font-size:1em; }
.resource-link { color:#FFD166; margin-right:.7em; }
#header {
  margin: 2.8em auto 1.7em auto; text-align: center;
}
#footer {
  color: #aaa; text-align: center; font-size:1.11em; margin-top: 3.5em;
}
hr { border: 0; border-top:1px solid #363a40;}
@media (max-width: 700px) {
  .result-card { font-size:1.04em; padding:1.2em 2vw; min-width: unset; }
  .prediction-header { font-size:1.3em; }
}
"""

# ------------------------------
# GRADIO APP DEFINITION
# ------------------------------
with gr.Blocks(css=CUSTOM_CSS, title="Disease Predictor") as demo:
    gr.HTML(
        """
        <div id='header'>
            <h1 style='color:#4EC3FA; font-weight:900; font-size: 2.5em; letter-spacing:-1.5px;'>Disease Prediction Assistant</h1>
            <div style='font-size:1.3em; color:#b9bec7;margin-top:1em;margin-bottom:1em;font-weight:500;'>
                Fast AI diagnosis from symptoms, plus info & remedies.<br>
                <span style="font-size:1em;color:#c9daad">Not medical advice. For education only.</span>
            </div>
        </div>"""
    )
    with gr.Row():
        symptom_input = gr.Dropdown(
            choices=all_symptoms,
            multiselect=True,
            label="Your Symptoms",
            info="Start typing and select all that apply"
        )
        predict_btn = gr.Button("🔎 Predict", elem_id="predict-btn", scale=0)
    output_html = gr.HTML()
    predict_btn.click(predict_disease, inputs=symptom_input, outputs=output_html)
    
    gr.HTML("<hr>")
    gr.Markdown("### 💬 Feedback (optional):")
    with gr.Row():
        feedback_box = gr.Textbox(lines=2, label="", placeholder="Your feedback, suggestions, or corrections...")
        feedback_btn = gr.Button("Send Feedback")
    feedback_out = gr.Markdown("")
    feedback_btn.click(handle_feedback, inputs=feedback_box, outputs=feedback_out)
    gr.HTML("<div id='footer'>Made with Gradio & SerpAPI &mdash; 2025</div>")

if __name__ == "__main__":
    demo.launch(share=False, show_error=True)