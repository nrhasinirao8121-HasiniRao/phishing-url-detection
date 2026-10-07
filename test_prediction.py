import os
import joblib
import pandas as pd
from feature_extraction import extract_features_df, extract_url_features, FEATURE_NAMES

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pkl")
META_PATH = os.path.join(BASE_DIR, "models", "feature_metadata.pkl")

model = joblib.load(MODEL_PATH)
metadata = joblib.load(META_PATH)

test_urls = [
    # Legitimate samples
    "https://www.google.com",
    "https://www.wikipedia.org",
    "https://www.southbankmosaics.com",
    "https://www.uni-mainz.de",
    # Phishing samples from dataset and real patterns
    "http://www.shprakserf.gq",
    "http://www.f0519141.xsph.ru",
    "https://service-mitld.firebaseapp.com/",
    "http://secure-login-account-update.tk/banking",
    "http://192.168.1.1/login-verify-account.php"
]

print("=" * 80)
print("TESTING PREDICTION PIPELINE ON SAMPLE URLS")
print("=" * 80)

df_feats = extract_features_df(test_urls, metadata=metadata)
predictions = model.predict(df_feats)
probabilities = model.predict_proba(df_feats)

for u, pred, prob in zip(test_urls, predictions, probabilities):
    label_name = "LEGITIMATE" if pred == 1 else "PHISHING"
    # Probability of the predicted class
    conf = prob[1] if pred == 1 else prob[0]
    print(f"URL: {u}")
    print(f"  -> Prediction : {label_name}")
    print(f"  -> Confidence : {conf * 100:.2f}% (P(Phish)={prob[0]*100:.1f}%, P(Legit)={prob[1]*100:.1f}%)")
    print("-" * 80)
