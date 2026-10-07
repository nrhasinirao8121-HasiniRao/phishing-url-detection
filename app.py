"""
app.py
======
Flask Web Application for:
AI-Based Phishing URL Detection Using Machine Learning

Provides a high-performance web dashboard with real-time URL feature extraction,
calibrated machine learning inference, and a cybersecurity operations theme.
"""

import os
import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_from_directory
from feature_extraction import extract_url_features, FEATURE_NAMES

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Load trained model and metadata
MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pkl")
META_PATH = os.path.join(MODELS_DIR, "feature_metadata.pkl")

model = None
metadata = None

try:
    if os.path.exists(MODEL_PATH) and os.path.exists(META_PATH):
        model = joblib.load(MODEL_PATH)
        metadata = joblib.load(META_PATH)
        print(f"[PhishGuard AI] Loaded model: {metadata.get('best_model_name', 'Random Forest')}")
    else:
        print("[WARNING] Model artifacts not found. Please run train_model.py first.")
except Exception as e:
    print(f"[ERROR] Failed to load model artifacts: {e}")


@app.route('/')
def home():
    """Renders the main cybersecurity operations dashboard."""
    metrics_summary = metadata.get('metrics_summary', []) if metadata else []
    best_model_name = metadata.get('best_model_name', 'Random Forest Classifier') if metadata else 'Random Forest'
    return render_template(
        'index.html',
        metrics=metrics_summary,
        best_model_name=best_model_name
    )


@app.route('/predict', methods=['POST'])
def predict():
    """
    Accepts URL string via JSON or form submission.
    Extracts features, runs ML model, and returns comprehensive prediction report.
    """
    if not model:
        return jsonify({
            'success': False,
            'error': 'Model artifacts not loaded. Please ensure train_model.py has run successfully.'
        }), 500

    # Handle JSON and Form Data
    if request.is_json:
        data = request.get_json()
        raw_url = data.get('url', '').strip()
    else:
        raw_url = request.form.get('url', '').strip()

    if not raw_url:
        return jsonify({
            'success': False,
            'error': 'Please provide a valid URL to analyze.'
        }), 400

    try:
        # 1. Feature Extraction
        features_dict = extract_url_features(raw_url, metadata=metadata)
        
        # 2. Format as single-row DataFrame with exact training column order
        df_input = pd.DataFrame([features_dict])[FEATURE_NAMES]

        # 3. Model Inference
        pred = int(model.predict(df_input)[0])
        prob = model.predict_proba(df_input)[0]

        prob_phish = float(prob[0])
        prob_legit = float(prob[1])

        # Meaning of labels: 1 = Legitimate, 0 = Phishing
        is_legitimate = (pred == 1)
        result_label = "LEGITIMATE" if is_legitimate else "PHISHING DETECTED"
        confidence_pct = round((prob_legit if is_legitimate else prob_phish) * 100, 2)

        # Risk Classification
        if not is_legitimate:
            risk_level = "CRITICAL / HIGH RISK" if prob_phish > 0.85 else "SUSPICIOUS / ELEVATED"
            status_color = "danger"
        else:
            risk_level = "LOW / VERIFIED SAFE" if prob_legit > 0.85 else "NEUTRAL / MODERATE"
            status_color = "success"

        # Key URL characteristics for user explanation
        extracted_highlights = {
            'URL Length': features_dict['URLLength'],
            'Domain Length': features_dict['DomainLength'],
            'Protocol': 'HTTPS (Encrypted)' if features_dict['IsHTTPS'] == 1 else 'HTTP (Unencrypted)',
            'Domain is IP': 'Yes (Suspicious)' if features_dict['IsDomainIP'] == 1 else 'No (Standard Domain)',
            'Subdomain Count': features_dict['NoOfSubDomain'],
            'Special Characters': features_dict['NoOfOtherSpecialCharsInURL'],
            'Obfuscated Characters': features_dict['NoOfObfuscatedChar'],
            'Letter Ratio': f"{features_dict['LetterRatioInURL'] * 100:.1f}%",
            'Digit Ratio': f"{features_dict['DegitRatioInURL'] * 100:.1f}%",
            'Char Continuation Rate': f"{features_dict['CharContinuationRate'] * 100:.1f}%",
            'TLD Legitimate Score': f"{features_dict['TLDLegitimateProb']:.4f}",
            'URL Similarity Index': f"{features_dict['URLSimilarityIndex']:.1f}"
        }

        response_payload = {
            'success': True,
            'url': raw_url,
            'prediction': result_label,
            'is_legitimate': is_legitimate,
            'confidence': confidence_pct,
            'prob_phishing': round(prob_phish * 100, 2),
            'prob_legitimate': round(prob_legit * 100, 2),
            'risk_level': risk_level,
            'status_color': status_color,
            'features': extracted_highlights,
            'raw_features': features_dict
        }

        # If standard form request, render template with result; else return JSON
        if not request.is_json and request.headers.get('X-Requested-With') != 'XMLHttpRequest':
            metrics_summary = metadata.get('metrics_summary', []) if metadata else []
            best_model_name = metadata.get('best_model_name', 'Random Forest') if metadata else 'Random Forest'
            return render_template(
                'index.html',
                metrics=metrics_summary,
                best_model_name=best_model_name,
                result=response_payload
            )

        return jsonify(response_payload)

    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Analysis failed: {str(e)}'
        }), 500


@app.route('/results/<path:filename>')
def serve_result_image(filename):
    """Serves generated evaluation visualization plots."""
    return send_from_directory(RESULTS_DIR, filename)


if __name__ == '__main__':
    print("\n" + "=" * 65)
    print("  [PhishGuard AI] Phishing URL Detection Server Starting")
    print("  URL: http://127.0.0.1:5000")
    print("=" * 65 + "\n")
    app.run(host='127.0.0.1', port=5000, debug=True)
