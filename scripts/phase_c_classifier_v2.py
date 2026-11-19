import json
import re
import ipaddress
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# ---------- Load data ----------
def load_jsonl(path):
    return [json.loads(l) for l in open(path)]

train = load_jsonl('working/working_dataset_train.jsonl')
test = load_jsonl('working/working_dataset_test.jsonl')

def get_prompt(r): return r['prompt']
def get_label(r): return r['completion'].strip()

def get_candidate(prompt):
    if ';' in prompt:
        return prompt.split(';', 1)[1].replace('->', '').strip()
    return prompt.strip()

X_train_raw = [get_prompt(r) for r in train]
X_test_raw  = [get_prompt(r) for r in test]
y_train = [get_label(r) for r in train]
y_test  = [get_label(r) for r in test]

# ---------- ML Model: TF-IDF (word n-grams) + Logistic Regression ----------
print("=== PHASE C: ML CLASSIFIER (TF-IDF + Logistic Regression) ===")
vectorizer = TfidfVectorizer(
    analyzer='word',
    ngram_range=(1, 3),
    token_pattern=r'[A-Za-z0-9:.]+',
    lowercase=False
)
X_train = vectorizer.fit_transform(X_train_raw)
X_test  = vectorizer.transform(X_test_raw)

model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("\n--- CLASSIFICATION REPORT (ML) ---")
print(classification_report(y_test, y_pred, zero_division=0))
ml_acc = accuracy_score(y_test, y_pred)

# ---------- Rule-based baseline ----------
print("\n=== RULE-BASED BASELINE (regex + ipaddress) ===")
ip_pattern = re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b')

def rule_based(prompt):
    cand = get_candidate(prompt)
    if not ip_pattern.search(cand):
        return 'negative'
    try:
        ipaddress.IPv4Address(cand)
        return 'positive'
    except ValueError:
        return 'negative'

y_base = [rule_based(p) for p in X_test_raw]
print("\n--- CLASSIFICATION REPORT (Baseline) ---")
print(classification_report(y_test, y_base, zero_division=0))
base_acc = accuracy_score(y_test, y_base)

# ---------- Majority baseline ----------
majority = max(set(y_train), key=y_train.count)
y_maj = [majority] * len(y_test)
maj_acc = accuracy_score(y_test, y_maj)
print(f"\nMajority-class baseline ('{majority}'): accuracy = {maj_acc:.4f}")

# ---------- Save predictions ----------
pred_df = pd.DataFrame({
    'prompt': X_test_raw,
    'candidate': [get_candidate(p) for p in X_test_raw],
    'actual': y_test,
    'ml_predicted': y_pred,
    'baseline_predicted': y_base
})
pred_df.to_csv('outputs/predictions.csv', index=False)

# ---------- Confusion matrix ----------
cm = confusion_matrix(y_test, y_pred, labels=['positive','negative'])
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['positive','negative'],
            yticklabels=['positive','negative'])
plt.title('ML Classifier Confusion Matrix')
plt.xlabel('Predicted'); plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('outputs/phase_c_confusion_matrix_v2.png')
plt.close()

# ---------- Clean holdout check (exclude 11 exact overlaps) ----------
train_prompts = set(X_train_raw)
clean_mask = [p not in train_prompts for p in X_test_raw]
clean_X = [X_test_raw[i] for i, keep in enumerate(clean_mask) if keep]
clean_y = [y_test[i] for i, keep in enumerate(clean_mask) if keep]
Xc = vectorizer.transform(clean_X)
yc_pred = model.predict(Xc)
clean_acc = accuracy_score(clean_y, yc_pred)

print(f"\n=== LEAKAGE-AWARE SUPPLEMENTARY EVAL ===")
print(f"Full test accuracy:  {ml_acc:.4f}  (n={len(y_test)})")
print(f"Clean-only accuracy: {clean_acc:.4f}  (n={len(clean_y)}, {len(y_test)-len(clean_y)} overlapping prompts removed)")
print(f"Rule baseline acc:   {base_acc:.4f}")
print(f"Majority baseline:   {maj_acc:.4f}")

# ---------- Write report ----------
with open('reports/phase_c_classifier_v2.txt', 'w') as f:
    f.write("=== PHASE C: OFFLINE CLASSIFIER (v2) ===\n\n")
    f.write("Model: TF-IDF (word n-grams 1-3) + Logistic Regression\n")
    f.write("Holdout: working_dataset_test.jsonl (n=46)\n\n")
    f.write("--- ML CLASSIFICATION REPORT ---\n")
    f.write(classification_report(y_test, y_pred, zero_division=0))
    f.write("\n--- RULE-BASED BASELINE REPORT ---\n")
    f.write(classification_report(y_test, y_base, zero_division=0))
    f.write(f"\nMajority baseline ('{majority}'): {maj_acc:.4f}\n")
    f.write(f"\nLeakage: 11 exact prompt overlaps between train and test\n")
    f.write(f"Full test accuracy:  {ml_acc:.4f}\n")
    f.write(f"Clean-only accuracy: {clean_acc:.4f} (n={len(clean_y)})\n")
    f.write(f"Rule baseline acc:   {base_acc:.4f}\n")

print("\nSaved: outputs/predictions.csv, outputs/phase_c_confusion_matrix_v2.png, reports/phase_c_classifier_v2.txt")
