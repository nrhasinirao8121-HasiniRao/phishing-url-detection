# Academic Mini-Project Report

## Title: AI-Based Phishing URL Detection Using Machine Learning
**Degree**: Bachelor of Technology (B.Tech) in Computer Science & Engineering (3rd Year)  
**Academic Domain**: Cybersecurity, Machine Learning & Web Technologies  

---

## Table of Contents
1. [Abstract](#1-abstract)
2. [Introduction](#2-introduction)
3. [Literature Survey](#3-literature-survey)
4. [Problem Statement](#4-problem-statement)
5. [Existing Methodology](#5-existing-methodology)
6. [Proposed Method](#6-proposed-method)
7. [Dataset Description](#7-dataset-description)
8. [System Architecture](#8-system-architecture)
9. [Methodology](#9-methodology)
10. [Implementation Details](#10-implementation-details)
11. [Results and Discussion](#11-results-and-discussion)
12. [Advantages](#12-advantages)
13. [Limitations](#13-limitations)
14. [Future Scope](#14-future-scope)
15. [Conclusion](#15-conclusion)
16. [References](#16-references)

---

## 1. Abstract
Phishing attacks continue to constitute one of the most persistent and damaging vectors of cyber exploitation, leveraging social engineering to deceive users into disclosing sensitive credentials, authentication tokens, and financial data. Conventional defensive countermeasures, such as static URL blacklists, suffer from inherent latency and fail to mitigate zero-day, fast-flux, or ephemeral phishing campaigns that operate for fewer than 4 to 8 hours. Conversely, dynamic content analysis methods requiring real-time webpage scraping introduce severe network overhead, latency, and vulnerabilities to malicious payloads.

This mini-project designs, implements, and evaluates an end-to-end Machine Learning phishing detection architecture utilizing the Kaggle **PhiUSIIL Phishing URL Dataset** comprising 235,795 real-world URLs. The system isolates 22 discriminative lexical, syntactic, and structural URL features that can be extracted instantaneously from raw URL strings without accessing remote web servers. Three classification algorithms—Logistic Regression, Decision Tree, and Random Forest—are benchmarked under an 80/20 stratified train/test split (47,074 unseen test samples). The optimal model, **Random Forest**, achieves **99.99% accuracy**, **100.00% precision**, and an **F1-score of 99.99%**. The trained inference pipeline is integrated into a full-stack Flask web application characterized by a dark cybersecurity operations dashboard, calibrated class posterior probabilities, and explainable feature breakdowns.

---

## 2. Introduction
The modern web ecosystem is critically reliant on Uniform Resource Locators (URLs) for resource addressing and inter-domain communication. Due to the ubiquity of hyperlinks in emails, SMS messages, and social media platforms, cybercriminals frequently exploit URL structures to mount deceptive attacks known as phishing. Attackers employ sophisticated evasion tactics, including:
- **Typo-squatting and Homoglyphs**: Crafting domain names visually indistinguishable from legitimate brands (e.g., substituting lowercase 'l' with numeral '1').
- **Excessive Subdomain Nesting**: Appending trusted organization names as subdomains of attacker-controlled root domains (e.g., `paypal.com.account-update.xyz`).
- **Percent-Encoding and Obfuscation**: Utilizing hexadecimal escapes to bypass simple pattern matching and signature detection.
- **Direct IP Address Hosting**: Bypassing Domain Name System (DNS) registration mechanisms and domain reputation filters.

To counteract these dynamic threats, supervised machine learning enables statistical generalization over non-linear patterns present in URL syntax. By analyzing lexical and structural signatures, machine learning models can identify deceptive intent prior to initiating network connections, offering proactive security directly at the endpoint.

---

## 3. Literature Survey

The academic literature on phishing detection is broadly categorized into blacklist-based systems, heuristic content-scraping frameworks, and machine-learning-driven lexical classifiers:

### 3.1 Lexical and Heuristic URL Analysis
*Sahoo, Liu, and Hoi (2017)* conducted a comprehensive survey on malicious URL detection using machine learning. They established that URL lexical features—such as path length, host length, delimiter distributions, and token entropy—provide strong discriminatory signals against cyber threats while circumventing the resource-intensive overhead of downloading and executing webpage code.

*Marchal et al. (2017)* presented "Off-the-Hook," an efficient client-side anti-phishing application that extracts lexical, DNS, and structural metrics. Their study proved that lexical URL features are immune to webpage evasion tactics, such as rendering login forms using obfuscated JavaScript canvas or CSS background images, which typically blind content-based scrapers.

### 3.2 Machine Learning & Ensemble Architectures
*Al-Sarem et al. (2021)* investigated feature selection and ensemble classification for phishing detection across multiple benchmark corpora. Their experimental findings demonstrated that ensemble tree-based models, particularly Random Forest, consistently outperform single-tree classifiers (Decision Trees) and linear classifiers (Logistic Regression) due to their resilience against variance and multicollinearity among correlated syntactic features.

*Hannousse and Yahiouche (2021)* performed an exhaustive experimental analysis of benchmark datasets for website phishing detection. They underscored the pervasive issue of data leakage in older benchmark datasets (such as the UCI Phishing Websites dataset), where synthetic or hand-crafted features led to inflated accuracy scores that deteriorated when applied to modern wild web traffic.

### 3.3 The PhiUSIIL Benchmark Dataset
*Prasad and Chandra (2023)* published the **PhiUSIIL Phishing URL Dataset** in *Computers & Security*. Addressing the limitations of legacy datasets, PhiUSIIL provides an updated, balanced collection of 235,795 URLs annotated with 55 distinct attributes spanning lexical indicators, character frequency probabilities, and webpage features. The authors demonstrated that character continuation rates, URL similarity indices, and delimiter frequencies are among the most robust indicators of deceptive web links.

---

## 4. Problem Statement
To engineer, train, evaluate, and deploy an automated, client-safe Machine Learning system capable of classifying website URLs as either **Legitimate** or **Phishing** in real time, fulfilling the following constraints:
1. Exclusively utilize the authentic **PhiUSIIL Phishing URL Dataset** without external data substitution.
2. Formulate a logically valid feature extraction pipeline that computes all model inputs directly from raw URL strings without requiring remote HTTP network calls.
3. Compare at least three classification algorithms using standard performance metrics.
4. Deploy the serialized model within a production-ready, dark cybersecurity-themed Flask web application with real-time prediction and feature explanation.

---

## 5. Existing Methodology
Current defensive mechanisms deployed in commodity web browsers exhibit critical limitations:

```
+-----------------------------------------------------------------------------+
|                            EXISTING METHODOLOGIES                           |
+-----------------------------------------------------------------------------+
| 1. Blacklist Systems (e.g., Google Safe Browsing, PhishTank)                |
|    - Drawback: Fails against zero-day URLs and rapid-flux domains.          |
|    - Latency: Hours to days before malicious URLs are reported and synced.  |
+-----------------------------------------------------------------------------+
| 2. Content-Based HTML/DOM Scrapers                                          |
|    - Drawback: Requires live HTTP GET requests; high network latency (>5s). |
|    - Vulnerability: Exposes scanning server to malicious malware payloads.  |
|    - Failure Mode: Cannot evaluate dead, taken-down, or rate-limited sites.  |
+-----------------------------------------------------------------------------+
| 3. Heuristic Rule-Based Filters                                             |
|    - Drawback: High false-positive rate; easily bypassed by obfuscation.    |
+-----------------------------------------------------------------------------+
```

---

## 6. Proposed Method
The proposed architecture circumvents the deficiencies of existing systems by implementing a **purely client-safe lexical machine-learning pipeline**:
1. **Zero Network Scraping**: Only the raw URL character string is evaluated. No HTTP connections, DNS queries, or socket connections are initiated toward the target domain.
2. **Deterministic Feature Engineering**: Extracts 22 mathematical and syntactic properties from the URL string, mirroring the construction parameters of the PhiUSIIL dataset.
3. **Calibrated Machine Learning Inference**: Uses an ensemble Random Forest model trained on 188,296 training records to output posterior probabilities $P(\text{Phishing} \mid X)$ and $P(\text{Legitimate} \mid X)$.
4. **Sub-10ms Inference**: Generates complete threat verdicts and feature breakdowns in under 10 milliseconds per URL.

---

## 7. Dataset Description

### 7.1 Dataset Overview
- **Name**: PhiUSIIL Phishing URL Dataset
- **Primary Source**: Kaggle / University of California Irvine (UCI) Machine Learning Repository
- **Reference**: Prasad & Chandra (Computers & Security, 2023)
- **Total Instances**: 235,795 URLs
- **Total Attributes**: 55 features

### 7.2 Class Distribution & Target Meaning
Through empirical verification and alignment with the published research paper, the target variable `label` is defined as:
- **`label = 1`**: **Legitimate Website** (134,850 samples, 57.29% post-deduplication)
- **`label = 0`**: **Phishing Website** (100,520 samples, 42.71% post-deduplication)

```
Target Label Breakdown:
  - Legitimate (Class 1): 134,850 samples [57.29%]
  - Phishing   (Class 0): 100,520 samples [42.71%]
  - Total Deduplicated  : 235,370 samples [100.0%]
```

### 7.3 Feature Categorization & Preprocessing Rationale
The 55 attributes of the dataset were analyzed and divided into three operational categories:
1. **Identifier / Data Leakage Columns (Excluded)**:
   - `URL`, `Domain`, `TLD`, `Title`: Raw strings with high cardinality that cannot generalize or introduce direct label memorization.
2. **Webpage Content Features (28 attributes — Excluded from Deployment)**:
   - `LineOfCode`, `NoOfiFrame`, `HasTitle`, `HasFavicon`, `Robots`, `IsResponsive`, `HasPasswordField`, `NoOfImage`, etc.
   - *Rationale*: Requiring these features during inference would force the web server to execute live HTML scraping of malicious URLs, introducing network latency, security hazards, and complete failure on inactive phishing domains.
3. **URL-Derived Lexical Features (22 attributes — Selected for Training & Production)**:
   - `URLLength`, `DomainLength`, `IsDomainIP`, `TLDLength`, `NoOfSubDomain`, `HasObfuscation`, `NoOfObfuscatedChar`, `ObfuscationRatio`, `NoOfLettersInURL`, `LetterRatioInURL`, `NoOfDegitsInURL`, `DegitRatioInURL`, `NoOfEqualsInURL`, `NoOfQMarkInURL`, `NoOfAmpersandInURL`, `NoOfOtherSpecialCharsInURL`, `SpacialCharRatioInURL`, `IsHTTPS`, `CharContinuationRate`, `URLCharProb`, `TLDLegitimateProb`, `URLSimilarityIndex`.

---

## 8. System Architecture

The end-to-end architecture is illustrated in the following workflow diagram:

```
[ User Input URL ]
        |
        v
[ Flask Web Server (app.py) ]
        |
        v
[ Feature Extraction Engine (feature_extraction.py) ]
   ├── URL Parsing (Scheme, Host, TLD, Subdomains, Query)
   ├── Lexical & Syntactic Feature Computation (Lengths, Ratios, Delimiters)
   ├── Obfuscation & Character Distribution Analysis
   └── Training-Learned Probability Lookup (TLD & Char Probability)
        |
        v
[ Vector Representation (22 Features Ordered Array) ]
        |
        v
[ Random Forest Classification Engine (best_model.pkl) ]
   ├── Evaluates 100 Decorrelated Decision Trees
   └── Computes Posterior Probabilities P(Phish) & P(Legit)
        |
        v
[ Decision & Risk Engine ]
   ├── Label: "LEGITIMATE" vs "PHISHING DETECTED"
   ├── Calibrated Confidence Score (%)
   └── Explainable Feature Breakdown
        |
        v
[ Cybersecurity Operations UI (Dashboard View) ]
```

---

## 9. Methodology

### 9.1 Data Cleansing and Preparation
1. **Handling Missing Values**: The dataset was audited for null or NaN entries across all 55 columns; 0 missing values were found.
2. **Deduplication**: 425 duplicate URL records were identified and dropped, yielding 235,370 clean rows.
3. **Train-Test Split**: Stratified random sampling with an 80/20 ratio and `random_state=42`:
   - Training Set: 188,296 samples
   - Testing Set: 47,074 samples

### 9.2 Machine Learning Classification Algorithms
Three diverse algorithms were implemented:

1. **Logistic Regression**:
   - A generalized linear model utilizing the sigmoid link function:
     $$\sigma(z) = \frac{1}{1 + e^{-z}}, \quad z = w^T x + b$$
   - Features normalized using `StandardScaler` ($\mu=0, \sigma=1$).
   - Optimization: L-BFGS solver with maximum iterations of 1000.
2. **Decision Tree Classifier**:
   - A non-parametric supervised learning algorithm utilizing the Gini impurity criterion:
     $$I_G(p) = 1 - \sum_{i=1}^{C} p_i^2$$
   - Maximum tree depth bounded at 16, minimum samples per split set to 10 to prevent overfitting.
3. **Random Forest Classifier**:
   - An ensemble bagging algorithm aggregating $B = 100$ decorrelated decision trees trained on bootstrap samples:
     $$\hat{y} = \text{mode}\left(\{T_b(x)\}_{b=1}^B\right)$$
   - Tree depth constrained to 20; all CPU cores utilized in parallel (`n_jobs=-1`).

---

## 10. Implementation Details

### 10.1 Key Source Files
- `feature_extraction.py`: Derives all 22 numerical features directly from raw input URLs with zero network dependencies.
- `train_model.py`: Automates data ingestion, deduplication, stratified splitting, model benchmarking, visualization plotting, and joblib artifact serialization.
- `app.py`: High-performance Flask server handling REST API routes and responsive HTML template rendering.
- `templates/index.html`: Modern cybersecurity operations center interface with dark aesthetics, hero input, sample testing pills, and modal chart inspection.
- `static/css/style.css`: Comprehensive design system featuring charcoal backgrounds, glassmorphism, glowing badges, and mobile-friendly responsive breakpoints.
- `static/js/app.js`: Dynamic interactive canvas rendering background cyber-network nodes and managing AJAX prediction requests.

---

## 11. Results and Discussion

### 11.1 Model Performance Comparison
All models were evaluated on the **47,074 unseen test samples**. The empirical metrics achieved were:

| Metric | Logistic Regression | Decision Tree | Random Forest |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 99.98% | 99.99% | **99.99%** |
| **Precision (Phishing)** | 100.00% | 99.98% | **100.00%** |
| **Recall (Phishing)** | 99.95% | 99.99% | **99.99%** |
| **F1-Score (Phishing)** | 99.98% | 99.98% | **99.99%** |
| **Macro F1-Score** | 99.98% | 99.98% | **99.99%** |
| **Training Time** | 0.29 seconds | 0.60 seconds | 3.89 seconds |

### 11.2 Detailed Evaluation of Random Forest (Champion Model)
The classification report for the final Random Forest model on 47,074 test records:

```
                precision    recall  f1-score   support

  Phishing (0)     1.0000    0.9999    0.9999     20,104
Legitimate (1)     0.9999    1.0000    0.9999     26,970

      accuracy                         0.9999     47,074
     macro avg     0.9999    0.9999    0.9999     47,074
  weighted avg     0.9999    0.9999    0.9999     47,074
```

### 11.3 Confusion Matrix
```
                         Predicted: Phishing (0)    Predicted: Legitimate (1)
Actual: Phishing (0)             20,101                        3
Actual: Legitimate (1)                0                   26,970
```
- **True Positives (Phishing correctly caught)**: 20,101
- **False Negatives (Phishing missed)**: 3
- **True Negatives (Legitimate passed)**: 26,970
- **False Positives (Legitimate flagged as phishing)**: 0

### 11.4 Feature Importance Analysis
Analysis of the Gini impurity reduction across all 100 ensemble trees revealed the top discriminative features:
1. `URLSimilarityIndex`: Highest individual contribution; identifies deceptive brand spoofing.
2. `IsHTTPS`: Differentiates opportunistic unencrypted phishing pages from secure platforms.
3. `SpacialCharRatioInURL`: Catches excessive punctuation and token delimiters.
4. `CharContinuationRate`: Quantifies continuous uninterrupted lexical spans.
5. `DegitRatioInURL`: Detects automated hash-like paths and IP tokenization.

---

## 12. Advantages
1. **Zero Runtime Network Dependency**: URLs are evaluated without requesting remote servers, preventing server compromise and network latency.
2. **Instantaneous Response Time**: Complete inference is executed in <10ms.
3. **High Discriminative Accuracy**: Achieves 99.99% accuracy on over 47,000 unseen test samples.
4. **Explainable AI**: Provides users with transparent metrics explaining why a given URL was flagged as malicious.
5. **Modern Operations Center Interface**: Professional dark cybersecurity UI suitable for technical demonstrations.

---

## 13. Limitations
1. **Shortened URL Redirection**: Shortened links (e.g. `bit.ly`, `tinyurl.com`) conceal original path structures until HTTP header unmasking is performed.
2. **Subdomain-Hosted Legitimate SaaS Abuse**: Free hosting providers (e.g., Firebase, GitHub Pages) may display legitimate base domain structures while hosting phishing landing pages.
3. **Domain Age Absence**: Lexical inspection alone cannot determine the registration age of a brand-new zero-day domain.

---

## 14. Future Scope
1. **Browser Extension Packaging**: Porting the feature extraction pipeline to WebExtensions (JavaScript) to inspect links directly in user browsers before clicks.
2. **DNS & WHOIS API Integration**: Supplementing lexical features with live domain age, registrar trust, and SSL validity checks.
3. **Deep Learning Hybrid Modeling**: Implementing Character-Level Bidirectional LSTMs and Convolutional Neural Networks (CNNs) to discover latent n-gram character sequences.

---

## 15. Conclusion
This project successfully designed, validated, and deployed an **AI-Based Phishing URL Detection System** using machine learning. By analyzing the 235,795 records of the benchmark **PhiUSIIL Phishing URL Dataset**, the project developed a logically sound 22-parameter feature extraction pipeline decoupled from unsafe web scraping. Benchmarking across Logistic Regression, Decision Trees, and Random Forest demonstrated the superior performance of Random Forest, achieving **99.99% accuracy** and an **F1-score of 99.99%**. The integrated Flask web application offers a modern, high-contrast cybersecurity dashboard providing instant threat classification and explainable feature breakdowns.

---

## 16. References

### 16.1 Dataset Reference
1. **Prasad, A., & Chandra, S. (2023).** "PhiUSIIL: A novel benchmark dataset for phishing URL detection using machine learning and deep learning techniques." *Computers & Security*, 135, 103545. [https://doi.org/10.1016/j.cose.2023.103545](https://doi.org/10.1016/j.cose.2023.103545)

### 16.2 Research Paper References
2. **Sahoo, D., Liu, C., & Hoi, S. C. (2017).** "Malicious URL detection using machine learning: A survey." *ACM Computing Surveys (CSUR)*, 50(6), 1–36.
3. **Marchal, S., Armano, G., Gröndahl, T., Saari, K., Singh, N., & Asokan, N. (2017).** "Off-the-hook: An efficient and usable client-side phishing prevention application." *IEEE Transactions on Computers*, 66(12), 2051–2065.
4. **Al-Sarem, M., Saeed, F., Boulila, W., Al-Hashida, G., & Al-Qaness, M. A. (2021).** "Feature selection and classification of phishing websites using ensemble machine learning techniques." *IEEE Access*, 9, 70365–70380.
5. **Hannousse, A., & Yahiouche, S. (2021).** "Towards benchmark datasets for machine learning based website phishing detection: An experimental study." *International Journal of Information Security*, 20(6), 771–797.
6. **Breiman, L. (2001).** "Random Forests." *Machine Learning*, 45(1), 5–32.

### 16.3 Software & Library References
7. **Pedregosa, F., et al. (2011).** "Scikit-learn: Machine Learning in Python." *Journal of Machine Learning Research*, 12, 2825–2830.
8. **McKinney, W. (2010).** "Data Structures for Statistical Computing in Python." *Proceedings of the 9th Python in Science Conference*, 51–56.
9. **Ronacher, A. (2010).** "Flask: A Python Microframework based on Werkzeug and Jinja 2." Pallets Projects.
