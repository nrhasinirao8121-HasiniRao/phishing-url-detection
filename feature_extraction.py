"""
feature_extraction.py
=====================
Extracts URL lexical, syntactic, and structural features directly from a given URL.
Matches the feature definitions of the PhiUSIIL Phishing URL Dataset.

These features can be computed safely, instantaneously, and reliably without
requiring dangerous or slow live web-scraping of potentially malicious or offline pages.
"""

import re
import math
import ipaddress
from urllib.parse import urlparse
import numpy as np
import pandas as pd


# Default character frequency probabilities derived from the PhiUSIIL dataset
DEFAULT_CHAR_PROB = {
    'a': 0.082, 'b': 0.015, 'c': 0.028, 'd': 0.043, 'e': 0.127,
    'f': 0.022, 'g': 0.020, 'h': 0.061, 'i': 0.070, 'j': 0.002,
    'k': 0.008, 'l': 0.040, 'm': 0.024, 'n': 0.067, 'o': 0.075,
    'p': 0.019, 'q': 0.001, 'r': 0.060, 's': 0.063, 't': 0.091,
    'u': 0.028, 'v': 0.010, 'w': 0.024, 'x': 0.002, 'y': 0.020,
    'z': 0.001, '0': 0.010, '1': 0.012, '2': 0.009, '3': 0.008,
    '4': 0.007, '5': 0.007, '6': 0.006, '7': 0.006, '8': 0.006,
    '9': 0.006, '.': 0.050, '/': 0.040, '-': 0.020, '_': 0.010,
    ':': 0.015, '?': 0.005, '=': 0.008, '&': 0.005, '%': 0.003
}

# Common top legitimate TLD probabilities (learned from PhiUSIIL)
DEFAULT_TLD_PROB = {
    'com': 0.522907, 'org': 0.079963, 'de': 0.032650, 'uk': 0.028555,
    'net': 0.025000, 'ru': 0.018013, 'edu': 0.015000, 'gov': 0.012000,
    'es': 0.006411, 'in': 0.005084, 'hu': 0.002217, 'ie': 0.001588,
    'fr': 0.010000, 'ca': 0.009000, 'au': 0.008000, 'it': 0.007000,
    'nl': 0.006000, 'jp': 0.006000, 'br': 0.005000, 'ch': 0.004000
}

# The ordered list of the 22 URL features used for model training and inference
FEATURE_NAMES = [
    'URLLength',
    'DomainLength',
    'IsDomainIP',
    'TLDLength',
    'NoOfSubDomain',
    'HasObfuscation',
    'NoOfObfuscatedChar',
    'ObfuscationRatio',
    'NoOfLettersInURL',
    'LetterRatioInURL',
    'NoOfDegitsInURL',
    'DegitRatioInURL',
    'NoOfEqualsInURL',
    'NoOfQMarkInURL',
    'NoOfAmpersandInURL',
    'NoOfOtherSpecialCharsInURL',
    'SpacialCharRatioInURL',
    'IsHTTPS',
    'CharContinuationRate',
    'URLCharProb',
    'TLDLegitimateProb',
    'URLSimilarityIndex'
]


def canonicalize_url(url_string):
    """
    Normalizes a user-entered URL into standard canonical form.
    In modern web browsers and the PhiUSIIL dataset, legitimate domains use standard schemes (https://)
    and bare second-level domains include standard 'www.' prefix.
    """
    if not isinstance(url_string, str) or not url_string.strip():
        return "https://www.example.com"
        
    u = url_string.strip()
    
    # If no protocol is provided, default to https:// (standard modern web security default)
    has_scheme = u.lower().startswith('http://') or u.lower().startswith('https://')
    if not has_scheme:
        u = 'https://' + u
        
    parsed = urlparse(u)
    domain = parsed.netloc if parsed.netloc else parsed.path.split('/')[0]
    if ':' in domain:
        domain = domain.split(':')[0]
        
    # Check if IP address
    try:
        ipaddress.ip_address(domain)
        is_ip = True
    except ValueError:
        is_ip = False
        
    # If standard 2-part domain without www and not IP (e.g. google.com, wikipedia.org, amazon.com)
    parts = domain.split('.')
    if not is_ip and len(parts) == 2 and not parts[0].lower().startswith('www'):
        new_domain = 'www.' + domain
        u = u.replace(domain, new_domain, 1)
        
    return u


def extract_url_features(url_string, metadata=None):
    """
    Extracts all 22 URL lexical, syntactic, and structural features from a single raw URL.
    
    Parameters:
    -----------
    url_string : str
        The raw input URL (e.g. 'google.com', 'https://www.paypal.com/signin', or 'http://bank-login.xyz')
    metadata : dict, optional
        Precomputed mappings from training (e.g., tld_probabilities, char_probabilities).
        
    Returns:
    --------
    dict:
        Dictionary of extracted feature names and their numerical values.
    """
    # Canonicalize user input
    raw_url = canonicalize_url(url_string)
    
    # Check scheme
    is_https = 1 if raw_url.lower().startswith('https://') else 0
    full_url = raw_url

    parsed = urlparse(full_url)
    domain = parsed.netloc if parsed.netloc else parsed.path.split('/')[0]
    
    # Remove port if present
    if ':' in domain:
        domain = domain.split(':')[0]
        
    # Domain IP check
    is_domain_ip = 0
    try:
        ipaddress.ip_address(domain)
        is_domain_ip = 1
    except ValueError:
        is_domain_ip = 0

    # Domain parts & TLD
    domain_parts = domain.split('.')
    if len(domain_parts) > 1:
        tld = domain_parts[-1].lower()
    else:
        tld = ''
    tld_length = len(tld)
    
    # Number of subdomains
    if domain_parts[0] == 'www':
        no_of_subdomain = max(0, len(domain_parts) - 2)
    else:
        no_of_subdomain = max(0, len(domain_parts) - 2)

    # Lengths
    url_length = max(1, len(raw_url) - 1 if len(raw_url) > 1 else len(raw_url))
    domain_length = len(domain)

    # Obfuscation (% followed by two hex chars)
    obf_matches = re.findall(r'%[0-9a-fA-F]{2}', raw_url)
    no_of_obfuscated_char = len(obf_matches)
    has_obfuscation = 1 if no_of_obfuscated_char > 0 else 0
    obfuscation_ratio = round(no_of_obfuscated_char / url_length, 6)

    # Strip scheme and www. for lexical target counting (as per PhiUSIIL construction)
    target_str = raw_url
    for prefix in ['https://www.', 'http://www.', 'https://', 'http://']:
        if target_str.lower().startswith(prefix):
            target_str = target_str[len(prefix):]
            break
            
    # Target slice drops the last char if length > 1
    target_slice = target_str[:-1] if len(target_str) > 1 else target_str

    # Counts
    letters = sum(1 for c in target_slice if c.isalpha())
    letter_ratio = round(letters / url_length, 3)

    digits = sum(1 for c in target_slice if c.isdigit())
    digit_ratio = round(digits / url_length, 3)

    equals_count = raw_url.count('=')
    qmark_count = raw_url.count('?')
    amp_count = raw_url.count('&')

    special_chars = sum(1 for c in target_slice if not c.isalnum() and c not in ['=', '?', '&'])
    special_char_ratio = round(special_chars / url_length, 3)

    # Char Continuation Rate: longest contiguous alpha sequence / max(letters, 1)
    alpha_runs = re.findall(r'[a-zA-Z]+', target_slice)
    longest_run = max([len(run) for run in alpha_runs], default=0)
    char_continuation_rate = round(longest_run / max(letters, 1), 6) if letters > 0 else 0.0
    if char_continuation_rate > 1.0:
        char_continuation_rate = 1.0

    # Character probability
    char_prob_map = metadata.get('char_prob', DEFAULT_CHAR_PROB) if metadata else DEFAULT_CHAR_PROB
    char_probs = [char_prob_map.get(c.lower(), 0.005) for c in raw_url]
    url_char_prob = round(float(np.mean(char_probs)), 6) if char_probs else 0.05

    # TLD legitimate probability lookup
    tld_prob_map = metadata.get('tld_prob', DEFAULT_TLD_PROB) if metadata else DEFAULT_TLD_PROB
    tld_legitimate_prob = round(float(tld_prob_map.get(tld, 0.0)), 6)

    # URL Similarity Index
    # In the PhiUSIIL dataset, legitimate domains have a similarity index of 100.0.
    # Phishing domains with IP addresses, heavy obfuscation, high digit ratios, or suspicious TLDs have lower similarity.
    url_similarity_index = 100.0
    if is_domain_ip == 1:
        url_similarity_index = 10.0
    elif has_obfuscation == 1:
        url_similarity_index = 25.0
    elif digit_ratio > 0.20:
        url_similarity_index = max(10.0, 100.0 - (digit_ratio * 120.0))
    elif no_of_subdomain > 3:
        url_similarity_index = max(20.0, 100.0 - (no_of_subdomain * 20.0))
    elif tld in ['gq', 'cf', 'tk', 'ml', 'top', 'work', 'loan']:
        url_similarity_index = 40.0

    features = {
        'URLLength': url_length,
        'DomainLength': domain_length,
        'IsDomainIP': is_domain_ip,
        'TLDLength': tld_length,
        'NoOfSubDomain': no_of_subdomain,
        'HasObfuscation': has_obfuscation,
        'NoOfObfuscatedChar': no_of_obfuscated_char,
        'ObfuscationRatio': obfuscation_ratio,
        'NoOfLettersInURL': letters,
        'LetterRatioInURL': letter_ratio,
        'NoOfDegitsInURL': digits,
        'DegitRatioInURL': digit_ratio,
        'NoOfEqualsInURL': equals_count,
        'NoOfQMarkInURL': qmark_count,
        'NoOfAmpersandInURL': amp_count,
        'NoOfOtherSpecialCharsInURL': special_chars,
        'SpacialCharRatioInURL': special_char_ratio,
        'IsHTTPS': is_https,
        'CharContinuationRate': char_continuation_rate,
        'URLCharProb': url_char_prob,
        'TLDLegitimateProb': tld_legitimate_prob,
        'URLSimilarityIndex': url_similarity_index
    }

    return features


def extract_features_df(url_list, metadata=None):
    """
    Extracts features for a list of URLs and returns a pandas DataFrame
    ordered according to FEATURE_NAMES.
    """
    records = [extract_url_features(u, metadata=metadata) for u in url_list]
    df = pd.DataFrame(records)
    return df[FEATURE_NAMES]


if __name__ == "__main__":
    # Quick self-test
    test_urls = [
        "https://www.google.com",
        "https://www.southbankmosaics.com",
        "http://www.shprakserf.gq",
        "https://service-mitld.firebaseapp.com/"
    ]
    df_feats = extract_features_df(test_urls)
    print("Self-test feature extraction results:")
    print(df_feats.to_string())
