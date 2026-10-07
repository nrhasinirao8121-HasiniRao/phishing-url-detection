import joblib
import pandas as pd
from urllib.parse import urlparse
import ipaddress
import re

model = joblib.load(r"C:\Users\Hasini Rao\.gemini\antigravity\scratch\phishing_url_detection\models\best_model.pkl")
meta = joblib.load(r"C:\Users\Hasini Rao\.gemini\antigravity\scratch\phishing_url_detection\models\feature_metadata.pkl")

from feature_extraction import extract_url_features, FEATURE_NAMES

def canonicalize_url(url_string):
    u = url_string.strip()
    # If no protocol is provided, default to https://
    has_scheme = u.lower().startswith('http://') or u.lower().startswith('https://')
    if not has_scheme:
        u = 'https://' + u
        
    parsed = urlparse(u)
    domain = parsed.netloc if parsed.netloc else parsed.path.split('/')[0]
    if ':' in domain:
        domain = domain.split(':')[0]
        
    # Check if IP
    try:
        ipaddress.ip_address(domain)
        is_ip = True
    except ValueError:
        is_ip = False
        
    # If standard 2-part domain without www and not IP (e.g. google.com, amazon.com, wikipedia.org)
    parts = domain.split('.')
    if not is_ip and len(parts) == 2 and not parts[0].lower().startswith('www'):
        # Add www.
        new_domain = 'www.' + domain
        u = u.replace(domain, new_domain, 1)
        
    return u

test_cases = [
    # Legitimate bare inputs:
    "google.com",
    "https://google.com",
    "www.google.com",
    "https://www.google.com",
    "wikipedia.org",
    "https://wikipedia.org",
    "github.com",
    "amazon.com",
    
    # Phishing inputs:
    "shprakserf.gq",
    "http://www.shprakserf.gq",
    "f0519141.xsph.ru",
    "http://www.f0519141.xsph.ru",
    "secure-login-account-update.tk/banking",
    "http://secure-login-account-update.tk/banking",
    "http://192.168.1.1/login.php",
    "192.168.1.1/login.php"
]

print(f"{'Input URL':45} | {'Canonical URL':45} | {'Pred':10} | {'P(Legit)':8} | {'P(Phish)':8}")
print("-" * 125)

for raw in test_cases:
    canon = canonicalize_url(raw)
    feats = extract_url_features(canon, metadata=meta)
    df_row = pd.DataFrame([feats])[FEATURE_NAMES]
    pred = model.predict(df_row)[0]
    prob = model.predict_proba(df_row)[0]
    label = "LEGIT" if pred == 1 else "PHISHING"
    print(f"{raw:45} | {canon:45} | {label:10} | {prob[1]*100:6.1f}% | {prob[0]*100:6.1f}%")
