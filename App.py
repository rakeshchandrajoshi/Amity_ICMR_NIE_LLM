
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import numpy as np

# Must be the first Streamlit command
st.set_page_config(page_title="Virus Prediction App", layout="centered")

# Load model and tokenizer
@st.cache_resource
def load_model_and_tokenizer():
    model_path = "Downloads/rrrr"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return tokenizer, model

tokenizer, model = load_model_and_tokenizer()

# Label mappings
unique_labels = [
    'Adeno Virus', 'Chikungunya', 'Coxsackie Virus', 'COVID-19', 'Cytomegalovirus (CMV)',
    'Dengue', 'Epstein-Barr-virus (EBV)', 'Hepatitis A virus (HAV)', 'Hepatitis B virus (HBV)',
    'Hepatitis C virus (HCV)', 'Herpes simplex virus (HSV)', 'Influenza  A', 'Influenza B',
    'Japanese Encephalitis', 'Leptospirosis', 'Measles', 'Mumps', 'Norovirus', 'Parvovirus B19',
    'Respiratory Syncytial Virus (RSV)', 'Rota Virus', 'Rubella Virus',
    'Orientia tsutsugamushi (Scrub typhus)', 'Varicella Zoaster Virus (VZV)', 'Zika',
    'Enterovirus', 'Coronavirus', 'Rhinovirus', 'Parainfluenza 1/2/3/4', 'Hepatitis E virus (HEV)', 'Parvovirus'
]
label2id = {label: i for i, label in enumerate(unique_labels)}
id2label = {i: label for label, i in label2id.items()}

# Prediction function
def predict_virus(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
        topk = torch.topk(logits, k=5, dim=-1)
        topk_indices = topk.indices[0].tolist()
        topk_scores = torch.nn.functional.softmax(logits, dim=-1)[0][topk_indices].tolist()
        predictions = [(id2label[idx], score) for idx, score in zip(topk_indices, topk_scores)]
        return predictions

# Example cases
example_cases = {
    "1. 48M, Maharashtra, respiratory symptoms": {
        "age": 48, "gender": "Male", "location": "Maharashtra",
        "symptoms": "respiratory sore, respiratory cough, respiratory breathlessness, respiratory fever"
    },
    "2. 8M, Jammu And Kashmir, fever": {
        "age": 8, "gender": "Male", "location": "Jammu And Kashmir",
        "symptoms": "fever"
    },
    "3. 2F, Kerala, rash papule": {
        "age": 2, "gender": "Female", "location": "Kerala",
        "symptoms": "rash papule"
    },
    "4. 9m, Assam, respiratory symptoms": {
        "age": 0.75, "gender": "Male", "location": "Assam",
        "symptoms": "respiratory sore, respiratory cough, respiratory breathlessness, respiratory fever"
    },
    "5. 60F, Tamil Nadu, diarrhoea fever, vomiting": {
        "age": 60, "gender": "Female", "location": "Tamil Nadu",
        "symptoms": "diarrhoea fever, diarrhoea vomiting"
    },
    "6. 1M, West Bengal, respiratory cough, fever": {
        "age": 1, "gender": "Male", "location": "West Bengal",
        "symptoms": "respiratory cough, respiratory fever"
    },
    "7. 5M, Bihar, jaundice symptoms": {
        "age": 5, "gender": "Male", "location": "Bihar",
        "symptoms": "jaundice fever, jaundice urine, jaundice hep"
    },
    "8. 21F, Punjab, fever": {
        "age": 21, "gender": "Female", "location": "Punjab",
        "symptoms": "fever"
    },
    "9. 6M, Odisha, encephalitis seizure": {
        "age": 6, "gender": "Male", "location": "Odisha",
        "symptoms": "encephalitis fever, encephalitis seizure"
    },
    "10. 16F, UP, encephalitis complex symptoms": {
        "age": 16, "gender": "Female", "location": "Uttar Pradesh",
        "symptoms": "encephalitis fever, encephalitis seizure, encephalitis sensorium, encephalitis somnelen, encephalitis irritab"
    },
}

# UI
st.title("🧪 AI Virus Prediction Assistant")

# Sample selector
st.subheader("🔍 Select a Sample Case (or fill manually below)")
selected_case = st.selectbox("Choose Example Case", ["-- None --"] + list(example_cases.keys()))

# Initialize with sample or default
default_data = {"age": 30, "gender": "Male", "location": "Kerala", "symptoms": "fever, cough"}
if selected_case != "-- None --":
    default_data = example_cases[selected_case]

# Input form
with st.form("patient_form"):
    age = st.number_input("Patient Age (in years)", min_value=0.0, max_value=120.0, value=float(default_data["age"]))
    gender = st.selectbox("Gender", ["Male", "Female"], index=0 if default_data["gender"] == "Male" else 1)
    location = st.text_input("Location (e.g., Tamil Nadu, Kerala)", value=default_data["location"])
    symptoms = st.text_area("Symptoms (comma-separated)", value=default_data["symptoms"])

    submitted = st.form_submit_button("🧬 Predict Virus")

if submitted:
    prompt = f"A {int(age)} years old {gender.lower()} from {location} with {symptoms} is likely infected with:"
    st.markdown(f"📄 **Prompt:** `{prompt}`")

    predictions = predict_virus(prompt)

    st.markdown("### 🎯 Top-5 Predicted Viruses")
    for i, (label, score) in enumerate(predictions, 1):
        st.markdown(f"**{i}. {label}** — Confidence: `{score:.2%}`")


