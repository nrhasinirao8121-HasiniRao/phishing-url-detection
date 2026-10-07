"""
train_model.py
==============
Machine Learning Training & Evaluation Pipeline for:
AI-Based Phishing URL Detection Using Machine Learning

Dataset: PhiUSIIL Phishing URL Dataset (Kaggle / Computers & Security 2023)
Author: 3rd-Year CSE Mini-Project
"""

import os
import sys
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Ensure results and models folders exist
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "PhiUSIIL_Phishing_URL_Dataset.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# URL features to select
URL_FEATURE_COLS = [
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

TARGET_COL = 'label'


def load_and_preprocess_data(dataset_path):
    """Loads and preprocesses the dataset."""
    print("=" * 75)
    print("STEP 1: LOADING AND INSPECTING DATASET")
    print("=" * 75)
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
        
    print(f"Loading dataset from: {dataset_path} ...")
    start_time = time.time()
    df = pd.read_csv(dataset_path)
    load_time = time.time() - start_time
    print(f"Loaded {len(df):,} rows and {df.shape[1]} columns in {load_time:.2f} seconds.")

    # Identify URL column (handle possible BOM '\ufeffURL')
    url_col = 'URL' if 'URL' in df.columns else [c for c in df.columns if 'URL' in c.upper()][0]

    # Check missing values
    missing_sum = df.isnull().sum().sum()
    print(f"Missing values across entire dataset: {missing_sum}")

    # Deduplication
    initial_count = len(df)
    df = df.drop_duplicates(subset=[url_col]).copy()
    dedup_removed = initial_count - len(df)
    print(f"Removed {dedup_removed:,} duplicate URL records. Remaining: {len(df):,} rows.")

    # Class distribution
    print("\nTarget Label Distribution:")
    counts = df[TARGET_COL].value_counts()
    for lbl, cnt in counts.items():
        name = "Legitimate" if lbl == 1 else "Phishing"
        pct = (cnt / len(df)) * 100
        print(f"  Class {lbl} ({name}): {cnt:,} samples ({pct:.2f}%)")

    # Plot Class Distribution
    plot_class_distribution(df[TARGET_COL])

    # Extract learned metadata for inference
    print("\nComputing training metadata for inference lookup...")
    tld_prob_map = df.groupby('TLD')['label'].mean().to_dict()
    
    # Save character frequency map
    char_counts = {}
    sample_urls = df[url_col].sample(n=min(50000, len(df)), random_state=42)
    total_chars = 0
    for u in sample_urls:
        for c in str(u).lower():
            char_counts[c] = char_counts.get(c, 0) + 1
            total_chars += 1
    char_prob_map = {c: cnt / total_chars for c, cnt in char_counts.items()}

    # Select features & target
    X = df[URL_FEATURE_COLS].copy()
    y = df[TARGET_COL].copy()

    # Verify no nulls in feature subset
    X = X.fillna(0)

    return X, y, tld_prob_map, char_prob_map


def plot_class_distribution(y_series):
    """Plots and saves class distribution chart."""
    plt.figure(figsize=(7, 5), dpi=300)
    sns.set_theme(style="darkgrid")
    
    counts = y_series.value_counts()
    labels = ['Legitimate (1)', 'Phishing (0)']
    values = [counts.get(1, 0), counts.get(0, 0)]
    colors = ['#10b981', '#ef4444']  # Green for Legitimate, Red for Phishing
    
    bars = plt.bar(labels, values, color=colors, width=0.5, edgecolor='#1e293b', linewidth=1.5)
    plt.title('Dataset Class Distribution (PhiUSIIL Dataset)', fontsize=13, fontweight='bold', pad=15)
    plt.ylabel('Number of Samples', fontsize=11)
    
    for bar in bars:
        height = bar.get_height()
        pct = (height / sum(values)) * 100
        plt.text(bar.get_x() + bar.get_width() / 2., height + 2000,
                 f'{height:,}\n({pct:.1f}%)',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.ylim(0, max(values) * 1.15)
    plt.tight_layout()
    chart_path = os.path.join(RESULTS_DIR, 'class_distribution.png')
    plt.savefig(chart_path)
    plt.close()
    print(f"Saved: {chart_path}")


def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """Trains 3 classification algorithms and evaluates their performance."""
    print("\n" + "=" * 75)
    print("STEP 2: TRAINING CLASSIFICATION ALGORITHMS")
    print("=" * 75)
    
    # Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        'Logistic Regression': (
            LogisticRegression(max_iter=1000, random_state=42),
            True  # Use scaled features
        ),
        'Decision Tree': (
            DecisionTreeClassifier(max_depth=16, min_samples_split=10, random_state=42),
            False # Trees are scale-invariant
        ),
        'Random Forest': (
            RandomForestClassifier(n_estimators=100, max_depth=20, n_jobs=-1, random_state=42),
            False
        )
    }

    results = {}
    trained_models = {}

    for name, (clf, use_scaled) in models.items():
        print(f"\nTraining [{name}]...")
        t0 = time.time()
        X_tr = X_train_scaled if use_scaled else X_train
        X_te = X_test_scaled if use_scaled else X_test
        
        clf.fit(X_tr, y_train)
        fit_time = time.time() - t0
        
        y_pred = clf.predict(X_te)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='binary', pos_label=0) # Phishing precision
        rec = recall_score(y_test, y_pred, average='binary', pos_label=0)    # Phishing recall
        f1 = f1_score(y_test, y_pred, average='binary', pos_label=0)        # Phishing F1
        
        # Macro metrics as well
        f1_macro = f1_score(y_test, y_pred, average='macro')

        results[name] = {
            'Accuracy': acc,
            'Precision (Phishing)': prec,
            'Recall (Phishing)': rec,
            'F1-Score (Phishing)': f1,
            'F1-Macro': f1_macro,
            'Training Time (s)': fit_time,
            'y_pred': y_pred
        }
        trained_models[name] = clf
        
        print(f" -> Completed in {fit_time:.2f}s | Accuracy: {acc*100:.2f}% | F1: {f1*100:.2f}%")

    # Generate Model Comparison Table
    print("\n" + "=" * 75)
    print("STEP 3: MODEL PERFORMANCE COMPARISON")
    print("=" * 75)
    
    summary_data = []
    for name, m in results.items():
        summary_data.append({
            'Model': name,
            'Accuracy': f"{m['Accuracy']*100:.2f}%",
            'Precision': f"{m['Precision (Phishing)']*100:.2f}%",
            'Recall': f"{m['Recall (Phishing)']*100:.2f}%",
            'F1-Score': f"{m['F1-Score (Phishing)']*100:.2f}%",
            'Training Time': f"{m['Training Time (s)']:.2f}s"
        })
    df_summary = pd.DataFrame(summary_data)
    print(df_summary.to_string(index=False))

    # Plot Model Comparison
    plot_model_comparison(results)

    # Determine Best Model (by F1-Score / Accuracy)
    best_model_name = max(results.keys(), key=lambda k: results[k]['F1-Score (Phishing)'])
    best_clf = trained_models[best_model_name]
    best_pred = results[best_model_name]['y_pred']

    print("\n" + "=" * 75)
    print(f"STEP 4: DETAILED EVALUATION OF BEST MODEL: {best_model_name.upper()}")
    print("=" * 75)
    
    # Classification Report
    target_names = ['Phishing (0)', 'Legitimate (1)']
    report_str = classification_report(y_test, best_pred, target_names=target_names, digits=4)
    print("Classification Report:")
    print(report_str)

    # Confusion Matrix
    cm = confusion_matrix(y_test, best_pred)
    print("Confusion Matrix:")
    print(cm)
    plot_confusion_matrix(cm, best_model_name)

    # Feature Importance (if tree model)
    if hasattr(best_clf, 'feature_importances_'):
        plot_feature_importance(best_clf, URL_FEATURE_COLS)

    return best_model_name, best_clf, scaler, results, df_summary, report_str, cm


def plot_model_comparison(results):
    """Generates grouped bar chart comparing all models."""
    plt.figure(figsize=(9, 5.5), dpi=300)
    sns.set_theme(style="darkgrid")
    
    model_names = list(results.keys())
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    bar_width = 0.2
    x = np.arange(len(model_names))
    
    palette = ['#3b82f6', '#10b981', '#f59e0b', '#8b5cf6']
    
    for i, metric in enumerate(metrics):
        metric_key = 'Accuracy' if metric == 'Accuracy' else f'{metric} (Phishing)'
        vals = [results[m][metric_key] * 100 for m in model_names]
        pos = x + (i - 1.5) * bar_width
        bars = plt.bar(pos, vals, width=bar_width, label=metric, color=palette[i], edgecolor='#1e293b')
        
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height + 0.8,
                     f'{height:.1f}%', ha='center', va='bottom', fontsize=8, rotation=90)

    plt.xticks(x, model_names, fontsize=11, fontweight='bold')
    plt.ylabel('Percentage (%)', fontsize=11)
    plt.title('Algorithm Comparison (Accuracy, Precision, Recall, F1)', fontsize=13, fontweight='bold', pad=15)
    plt.ylim(60, 105)
    plt.legend(loc='lower right', frameon=True)
    plt.tight_layout()
    chart_path = os.path.join(RESULTS_DIR, 'model_comparison.png')
    plt.savefig(chart_path)
    plt.close()
    print(f"Saved: {chart_path}")


def plot_confusion_matrix(cm, model_name):
    """Plots confusion matrix heatmap."""
    plt.figure(figsize=(6, 5), dpi=300)
    
    # Calculate percentages
    cm_sum = np.sum(cm)
    cm_pct = cm / cm_sum * 100
    labels = np.array([
        [f"{cm[0, 0]:,}\n({cm_pct[0, 0]:.1f}%)\n[True Phishing]", f"{cm[0, 1]:,}\n({cm_pct[0, 1]:.1f}%)\n[False Legitimate]"],
        [f"{cm[1, 0]:,}\n({cm_pct[1, 0]:.1f}%)\n[False Phishing]", f"{cm[1, 1]:,}\n({cm_pct[1, 1]:.1f}%)\n[True Legitimate]"]
    ])
    
    sns.heatmap(cm, annot=labels, fmt='', cmap='Blues', cbar=False,
                xticklabels=['Pred: Phishing', 'Pred: Legitimate'],
                yticklabels=['Actual: Phishing', 'Actual: Legitimate'],
                annot_kws={'fontsize': 10, 'fontweight': 'bold'})
                
    plt.title(f'Confusion Matrix - {model_name}', fontsize=12, fontweight='bold', pad=15)
    plt.ylabel('Ground Truth', fontsize=11)
    plt.xlabel('Predicted Label', fontsize=11)
    plt.tight_layout()
    chart_path = os.path.join(RESULTS_DIR, 'confusion_matrix.png')
    plt.savefig(chart_path)
    plt.close()
    print(f"Saved: {chart_path}")


def plot_feature_importance(clf, feature_names):
    """Plots top feature importances for tree-based models."""
    plt.figure(figsize=(8, 6), dpi=300)
    sns.set_theme(style="darkgrid")
    
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1][:15] # Top 15
    
    top_features = [feature_names[i] for i in indices]
    top_importances = [importances[i] * 100 for i in indices]
    
    # Horizontal bar plot
    y_pos = np.arange(len(top_features))
    plt.barh(y_pos, top_importances[::-1], color='#06b6d4', edgecolor='#0f172a')
    plt.yticks(y_pos, top_features[::-1], fontsize=10)
    plt.xlabel('Relative Importance (%)', fontsize=11)
    plt.title('Top 15 Most Discriminative URL Features', fontsize=13, fontweight='bold', pad=15)
    
    for i, val in enumerate(top_importances[::-1]):
        plt.text(val + 0.3, i, f'{val:.1f}%', va='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    chart_path = os.path.join(RESULTS_DIR, 'feature_importance.png')
    plt.savefig(chart_path)
    plt.close()
    print(f"Saved: {chart_path}")


def save_artifacts(best_name, best_clf, scaler, tld_prob_map, char_prob_map, results, df_summary):
    """Saves model, scaler, and metadata dictionaries."""
    print("\n" + "=" * 75)
    print("STEP 5: SERIALIZING MODEL AND PREPROCESSING ARTIFACTS")
    print("=" * 75)
    
    model_path = os.path.join(MODELS_DIR, "best_model.pkl")
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    meta_path = os.path.join(MODELS_DIR, "feature_metadata.pkl")

    # Save model and scaler
    joblib.dump(best_clf, model_path)
    joblib.dump(scaler, scaler_path)

    # Save metadata dictionary
    metadata = {
        'best_model_name': best_name,
        'feature_names': URL_FEATURE_COLS,
        'tld_prob': tld_prob_map,
        'char_prob': char_prob_map,
        'metrics_summary': df_summary.to_dict(orient='records'),
        'label_mapping': {1: 'Legitimate', 0: 'Phishing'},
        'trained_at': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    joblib.dump(metadata, meta_path)

    print(f"Saved Best Model ({best_name}) to: {model_path}")
    print(f"Saved Scaler to: {scaler_path}")
    print(f"Saved Feature Metadata to: {meta_path}")


def main():
    print("=" * 75)
    print("AI-BASED PHISHING URL DETECTION - MODEL TRAINING PIPELINE")
    print("=" * 75)
    
    # 1. Load data
    X, y, tld_prob_map, char_prob_map = load_and_preprocess_data(DATASET_PATH)

    # 2. Train-test split (80-20 stratified)
    print("\nSplitting dataset (80% Train, 20% Test, Stratified, random_state=42)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Training set: {X_train.shape[0]:,} samples")
    print(f"Testing set : {X_test.shape[0]:,} samples")

    # 3. Train and evaluate
    best_name, best_clf, scaler, results, df_summary, report_str, cm = train_and_evaluate_models(
        X_train, X_test, y_train, y_test
    )

    # 4. Save artifacts
    save_artifacts(best_name, best_clf, scaler, tld_prob_map, char_prob_map, results, df_summary)

    print("\n" + "=" * 75)
    print("TRAINING PIPELINE SUCCESSFULLY COMPLETED!")
    print("=" * 75)


if __name__ == "__main__":
    main()
