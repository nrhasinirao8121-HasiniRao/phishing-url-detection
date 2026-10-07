# AI-Based Phishing URL Detection Using Machine Learning

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.9.1-orange.svg)](https://scikit-learn.org/)
[![Flask](https://img.shields.io/badge/Flask-3.1.3-green.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/Project-3rd--Year%20CSE%20Mini--Project-purple.svg)]()

An end-to-end Machine Learning cybersecurity application designed to detect and classify phishing URLs in real-time using supervised machine learning algorithms trained on the **PhiUSIIL Phishing URL Dataset** (235,000+ real-world URLs).

---

## 1. Abstract
Phishing attacks remain one of the most prevalent and damaging attack vectors in modern cybersecurity, targeting human vulnerabilities rather than cryptographic weaknesses. Malicious actors disguise deceptive URLs as legitimate financial institutions, enterprise logins, and communication platforms to harvest user credentials and sensitive information. This project introduces an intelligent, machine-learning-driven phishing detection architecture that evaluates uniform resource locators (URLs) based on 22 lexical, syntactic, and structural features extracted directly from raw URL strings. By decoupling inference from runtime webpage scraping, the system delivers instantaneous (<10ms) threat classification with an accuracy of **99.99%** and an F1-score of **99.99%** using an optimized **Random Forest Classifier**.

---

## 2. Problem Statement
Traditional URL filtering mechanisms rely heavily on static blacklists (e.g., Google Safe Browsing, PhishTank). While effective for known threats, blacklists suffer from critical latency:
- Modern phishing campaigns have an average lifespan of under 4 to 8 hours.
- Attackers continuously generate new zero-hour domains, utilize fast-flux DNS, and deploy homograph spoofing.
- Blacklists fail completely against zero-day malicious domains.

Conversely, deep content-inspection systems that scrape live HTML pages suffer from severe latency (5–10 seconds per lookup), vulnerability to malicious payload execution, and immediate failure when phishing servers are temporarily suspended or unresponsive. There is an urgent need for a lightweight, client-safe, machine learning solution capable of detecting phishing URLs purely from structural URL syntax without accessing remote web servers.

---

## 3. Project Objectives
1. **Analyze and preprocess** the benchmark Kaggle **PhiUSIIL Phishing URL Dataset** (235,795 URLs).
2. **Reverse-engineer and implement** a self-contained feature extraction pipeline deriving 22 genuine URL lexical and syntactic characteristics from user-supplied URLs.
3. **Train and benchmark** three supervised classification algorithms:
   - Logistic Regression (Linear baseline)
   - Decision Tree Classifier (Non-linear rule tree)
   - Random Forest Classifier (Ensemble decision forest)
4. **Evaluate performance** using Accuracy, Precision, Recall, F1-Score, Confusion Matrices, and Feature Importance distributions.
5. **Develop a full-stack cybersecurity web dashboard** using Flask, HTML5, CSS3, and JavaScript featuring a modern Operations Center dark theme, live probability scoring, and explainable feature breakdowns.

---

## 4. Technologies Used
- **Programming Language**: Python 3.13
- **Data Engineering**: Pandas, NumPy
- **Machine Learning**: Scikit-Learn
- **Model Serialization**: Joblib
- **Data Visualization**: Matplotlib, Seaborn
- **Backend Framework**: Flask
- **Frontend / UI**: HTML5, Vanilla CSS3 (Custom Dark Cybersecurity Operations Theme), Modern JavaScript (Fetch API, HTML5 Canvas Network Animation)

---

## 5. Dataset Description
The project utilizes the **PhiUSIIL Phishing URL Dataset** ([Prasad & Chandra, Computers & Security, 2023](https://doi.org/10.1016/j.cose.2023.103545)):
- **Total Samples**: 235,795 URLs
- **Total Columns**: 55 features
- **Missing Values**: 0 missing values across all features
- **Duplicates Removed**: 425 duplicate URL strings
- **Target Variable (`label`)**:
  - `1`: **Legitimate** (134,850 samples — 57.29%)
  - `0`: **Phishing** (100,520 samples — 42.71%)
- **Feature Categories**:
  - *Leakage / Identifier Columns*: `URL`, `Domain`, `TLD`, `Title` (excluded from model training).
  - *Webpage Content Features (28)*: `LineOfCode`, `NoOfiFrame`, `HasTitle`, `NoOfImage`, etc. (excluded from inference to maintain safety and speed).
  - *URL-Derived Features (22)*: Lexical, structural, and character-frequency indicators used for model training and live deployment.

---

## 6. Machine Learning Algorithms & Results

The models were evaluated on an **80/20 stratified train/test split** consisting of **47,074 unseen test URLs**:

| Model | Accuracy | Precision (Phishing) | Recall (Phishing) | F1-Score (Phishing) | Training Time |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 99.98% | 100.00% | 99.95% | 99.98% | 0.29s |
| **Decision Tree** | 99.99% | 99.98% | 99.99% | 99.98% | 0.60s |
| **Random Forest (Champion)** | **99.99%** | **100.00%** | **99.99%** | **99.99%** | **3.89s** |

### Confusion Matrix (Random Forest on 47,074 Test URLs)
```
                   Predicted Phishing    Predicted Legitimate
Actual Phishing:        20,101                    3
Actual Legitimate:           0               26,970
```
- **False Positives**: Only 3 legitimate predictions out of 20,104 true phishing URLs.
- **False Negatives**: 0 false negatives on legitimate traffic.

All generated visualization graphs are automatically stored in the `results/` folder:
- `results/class_distribution.png`
- `results/model_comparison.png`
- `results/confusion_matrix.png`
- `results/feature_importance.png`

---

## 7. Project Structure
```
phishing_url_detection/
├── dataset/
│   └── PhiUSIIL_Phishing_URL_Dataset.csv   # Primary benchmark dataset
├── models/
│   ├── best_model.pkl                      # Serialized Random Forest classifier
│   ├── scaler.pkl                          # Fitted StandardScaler
│   └── feature_metadata.pkl                # Learned probability maps & metadata
├── results/
│   ├── class_distribution.png              # Class balance plot
│   ├── model_comparison.png                # Accuracy & F1 comparison bar chart
│   ├── confusion_matrix.png                # Best model confusion matrix heatmap
│   └── feature_importance.png              # Top discriminative features plot
├── templates/
│   └── index.html                          # Cybersecurity Operations web UI
├── static/
│   ├── css/
│   │   └── style.css                       # Dark cybersecurity theme styles
│   └── js/
│       └── app.js                          # Canvas animations & AJAX analyzer
├── feature_extraction.py                   # 22-parameter URL feature extractor
├── train_model.py                          # Training, benchmarking, and saving pipeline
├── app.py                                  # Flask web server
├── test_prediction.py                      # Standalone CLI testing script
├── requirements.txt                        # Python dependencies
├── README.md                               # Project documentation
└── project_report.md                       # Comprehensive 3rd-year CSE academic report
```

---

## 8. Installation & Setup Instructions

### Prerequisites
- Python 3.10+ (tested and verified on Python 3.13)
- Git (optional)

### Step 1: Clone or Open Project Directory
```powershell
cd "C:\Users\Hasini Rao\.gemini\antigravity\scratch\phishing_url_detection"
```

### Step 2: Install Dependencies
```powershell
pip install -r requirements.txt
```

### Step 3: Train the Machine Learning Model
Run the complete training pipeline to process the dataset, evaluate models, generate charts, and serialize the trained model:
```powershell
python train_model.py
```

### Step 4: Verify with CLI Test Script
Test sample legitimate and phishing URLs directly via terminal:
```powershell
python test_prediction.py
```

### Step 5: Launch the Web Application
Start the local Flask development server:
```powershell
python app.py
```
Open your browser and navigate to:
```
http://127.0.0.1:5000
```

---

## 9. Web Application Features
- **Cybersecurity Operations Theme**: Charcoal background (`#070a13`), emerald green legitimate accents, ruby red warning badges, and cyan highlights.
- **Dynamic Network Canvas**: Subtle animated particle node graph rendering in the background.
- **Asynchronous Analysis**: Instantaneous evaluation using the Fetch API without full page reloads.
- **Calibrated Probability Meter**: Visual comparison of posterior probabilities for both Phishing and Legitimate classes.
- **Explainable URL Characteristics**: Interactive grid displaying 12 key extracted features for every URL analyzed.
- **One-Click Test Samples**: Pre-configured buttons to test verified Legitimate and Phishing domains during academic demonstrations.
- **Evaluation Chart Lightbox**: Click any visualization chart in the "Model Performance" section to open a high-resolution modal.

---

## 10. Future Scope
- **Browser Extension Integration**: Packaging the URL feature extraction pipeline as a Chrome/Firefox extension for real-time URL vetting before page navigation.
- **Deep Learning / Transformer Models**: Incorporating 1D-CNN or character-level Bidirectional LSTM models for sequential URL character modeling.
- **DNS & WHOIS Enrichment**: Querying domain age, registrar reputation, and SSL certificate issuer dynamically for enhanced threat intelligence.
