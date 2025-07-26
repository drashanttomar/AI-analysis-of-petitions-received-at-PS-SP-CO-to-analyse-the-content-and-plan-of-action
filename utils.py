import joblib
import numpy as np
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer

base_path = os.path.dirname(__file__)

# Load models
claim_model = joblib.load(os.path.join(base_path, "models", "claim_classifier.pkl"))
claim_vectorizer = joblib.load(os.path.join(base_path, "models", "claim_vectorizer.pkl"))

score_model = joblib.load(os.path.join(base_path, "models", "score_predictor.pkl"))
score_vectorizer = joblib.load(os.path.join(base_path, "models", "score_vectorizer.pkl"))

status_model = joblib.load(os.path.join(base_path, "models", "status_classifier.pkl"))
status_vectorizer = joblib.load(os.path.join(base_path, "models", "status_vectorizer.pkl"))

def preprocess_text(text):
    """
    Preprocesses the input text by:
    - Converting to lowercase
    - Removing non-alphanumeric characters
    - Removing extra whitespace
    """
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\\s]', '', text)
    text = re.sub(r'\\s+', ' ', text).strip()
    return text

def predict_claim_type(text):
    vec = claim_vectorizer.transform([text])
    return claim_model.predict(vec)[0]

def predict_score(response):
    vec = score_vectorizer.transform([response])
    return score_model.predict(vec)[0]

def predict_status(complaint, claim, response, score):
    combined = f"{complaint} {claim} {response}"
    vec = status_vectorizer.transform([combined]).toarray()
    final_input = np.hstack((vec, [[score]]))
    return status_model.predict(final_input)[0]

def get_guidance(complaint, claim, response):
    claim = claim.lower()

    if "missing" in claim:
        guidance = (
            "- Expand Call Detail Records (CDR) to include past week.\n"
            "- Verify financial transactions or ATM usage.\n"
            "- Retrieve last seen location from telecom service provider.\n"
            "- Involve cyber cell for social media tracking.\n"
            "- Coordinate with nearby districts and share details."
        )
    elif "fraud" in claim or "cyber" in claim:
        guidance = (
            "- Request bank transaction logs and freeze suspected accounts.\n"
            "- Trace caller's location using CDR and IMEI.\n"
            "- Contact cyber crime unit for digital trail analysis.\n"
            "- Verify IP logs from telecom operator.\n"
            "- Check for previous complaints on the same number/account."
        )
    elif "land" in claim:
        guidance = (
            "- Cross-check land registry and ownership records.\n"
            "- Contact Patwari and revenue department for demarcation.\n"
            "- Prevent escalation by enforcing temporary injunction.\n"
            "- Ensure both parties are legally notified.\n"
            "- Recommend civil court proceedings if documents are disputed."
        )
    elif "domestic" in claim:
        guidance = (
            "- Record victim and witness statements sensitively.\n"
            "- Conduct a medical examination with proper documentation.\n"
            "- Ensure victim safety through immediate protection measures.\n"
            "- Involve women’s help cell if required.\n"
            "- Register FIR under relevant IPC and Domestic Violence Act."
        )
    else:
        guidance = (
            "- Review complaint content thoroughly.\n"
            "- Ensure FIR has been registered if needed.\n"
            "- Contact relevant units for support (e.g., cyber cell, legal).\n"
            "- Update complainant about current status regularly.\n"
            "- Submit progress report to supervising officer."
        )

    return guidance
