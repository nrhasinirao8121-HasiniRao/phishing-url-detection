import joblib
import pandas as pd
from feature_extraction import extract_features_df, extract_url_features

model = joblib.load(r"C:\Users\Hasini Rao\.gemini\antigravity\scratch\phishing_url_detection\models\best_model.pkl")
meta = joblib.load(r"C:\Users\Hasini Rao\.gemini\antigravity\scratch\phishing_url_detection\models\feature_metadata.pkl")

for u in ['google.com', 'https://www.google.com', 'https://google.com', 'http://google.com']:
    f = extract_features_df([u], metadata=meta)
    p = model.predict(f)[0]
    prob = model.predict_proba(f)[0]
    print(f"{u:25} -> pred={p} | P(Phish)={prob[0]*100:.1f}% | P(Legit)={prob[1]*100:.1f}% | is_https={f['IsHTTPS'].values[0]} | url_len={f['URLLength'].values[0]}")
