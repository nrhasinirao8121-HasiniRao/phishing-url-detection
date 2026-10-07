import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from app import app

client = app.test_client()

print("=" * 70)
print("TESTING FLASK APPLICATION ENDPOINTS")
print("=" * 70)

# Test GET /
res_home = client.get('/')
print(f"GET / : Status {res_home.status_code}")
assert res_home.status_code == 200, "Home page returned non-200"
assert b"AI-Based Phishing URL Detection" in res_home.data, "Title not found on home page"
print(" -> Home page loaded with correct HTML and titles.")

# Test POST /predict with Legitimate URL
legit_url = "https://www.google.com"
res_legit = client.post('/predict', json={'url': legit_url})
print(f"\nPOST /predict (Legitimate: {legit_url}) : Status {res_legit.status_code}")
data_legit = res_legit.get_json()
print("  Response:", data_legit['prediction'], f"Confidence: {data_legit['confidence']}%")
assert data_legit['success'] is True
assert data_legit['is_legitimate'] is True

# Test POST /predict with Phishing URL
phish_url = "http://www.shprakserf.gq"
res_phish = client.post('/predict', json={'url': phish_url})
print(f"\nPOST /predict (Phishing: {phish_url}) : Status {res_phish.status_code}")
data_phish = res_phish.get_json()
print("  Response:", data_phish['prediction'], f"Confidence: {data_phish['confidence']}%")
assert data_phish['success'] is True
assert data_phish['is_legitimate'] is False

# Test static results image serving
res_img = client.get('/results/model_comparison.png')
print(f"\nGET /results/model_comparison.png : Status {res_img.status_code}, Length: {len(res_img.data)} bytes")
assert res_img.status_code == 200

print("\n" + "=" * 70)
print("ALL FLASK ENDPOINT TESTS PASSED SUCCESSFULLY!")
print("=" * 70)
