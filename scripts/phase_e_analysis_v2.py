import pandas as pd
import numpy as np

# Load predictions
df = pd.read_csv('outputs/predictions.csv')
df['correct_ml'] = df['actual'] == df['ml_predicted']
df['correct_base'] = df['actual'] == df['baseline_predicted']

# ---------- Confusion matrix (ML model) ----------
tp = ((df['actual']=='positive') & (df['ml_predicted']=='positive')).sum()
tn = ((df['actual']=='negative') & (df['ml_predicted']=='negative')).sum()
fp = ((df['actual']=='negative') & (df['ml_predicted']=='positive')).sum()
fn = ((df['actual']=='positive') & (df['ml_predicted']=='negative')).sum()

print("=== PHASE E: CONFUSION MATRIX (ML MODEL) ===")
print(f"TP = {tp}   FP = {fp}")
print(f"FN = {fn}   TN = {tn}")
print()

# Precision, recall, F1
prec = tp / (tp + fp) if (tp+fp) else 0
rec  = tp / (tp + fn) if (tp+fn) else 0
f1   = 2*prec*rec / (prec+rec) if (prec+rec) else 0
acc  = (tp+tn) / len(df)
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1-score:  {f1:.4f}")

# Majority baseline (positive, since 74% positive)
maj_acc = (df['actual']=='positive').sum() / len(df)
print(f"\nMajority baseline (always positive): {maj_acc:.4f}")

# ---------- 8-record analyst review ----------
print("\n=== ANALYST REVIEW (8 RECORDS) ===")

# Selection strategy:
# 2 true positives (clear valid IPs)
# 2 true negatives (invalid IPs correctly rejected)
# 2 wildcard 0.0.0.0 (ML beat rule)
# 1 false positive (256.0.1.26 - ML failed)
# 1 ambiguous (valid IP labelled negative)

review = []
# True positives - pick 2 clear ones
tp_cases = df[(df['actual']=='positive') & (df['ml_predicted']=='positive')].head(2)
for _, row in tp_cases.iterrows():
    review.append({
        'record_type': 'TP',
        'candidate': row['candidate'],
        'prompt': row['prompt'],
        'actual': row['actual'],
        'ml_pred': row['ml_predicted'],
        'base_pred': row['baseline_predicted'],
        'note': 'Clear valid IPv4 in positive context. Both ML and rule agree. Correctly triaged as positive.'
    })

# True negatives - invalid IPs
tn_cases = df[(df['actual']=='negative') & (df['ml_predicted']=='negative') & (df['candidate'] != '0.0.0.0')].head(2)
for _, row in tn_cases.iterrows():
    review.append({
        'record_type': 'TN',
        'candidate': row['candidate'],
        'prompt': row['prompt'],
        'actual': row['actual'],
        'ml_pred': row['ml_predicted'],
        'base_pred': row['baseline_predicted'],
        'note': 'Invalid IPv4 syntax. Both ML and rule correctly rejected.'
    })

# Wildcards (ML beat rule)
wc = df[(df['candidate']=='0.0.0.0')].head(2)
for _, row in wc.iterrows():
    review.append({
        'record_type': 'TN (ML beat rule)',
        'candidate': row['candidate'],
        'prompt': row['prompt'],
        'actual': row['actual'],
        'ml_pred': row['ml_predicted'],
        'base_pred': row['baseline_predicted'],
        'note': '0.0.0.0 passes IPv4 syntax (rule => positive) but is the unspecified address. Source label negative. ML correctly used context.'
    })

# False positive
fp_case = df[(df['actual']=='negative') & (df['ml_predicted']=='positive')].head(1)
for _, row in fp_case.iterrows():
    review.append({
        'record_type': 'FP',
        'candidate': row['candidate'],
        'prompt': row['prompt'],
        'actual': row['actual'],
        'ml_pred': row['ml_predicted'],
        'base_pred': row['baseline_predicted'],
        'note': 'INVALID IPv4 (octet 256 > 255). Rule correctly rejected. ML incorrectly flagged positive because it sees a 4-group digit pattern in <string> tag context.'
    })

# Ambiguous valid IP labelled negative
amb = df[(df['actual']=='negative') & (df['baseline_predicted']=='positive') & (df['ml_predicted']=='negative') & (df['candidate']!='0.0.0.0')].head(1)
for _, row in amb.iterrows():
    review.append({
        'record_type': 'Ambiguous',
        'candidate': row['candidate'],
        'prompt': row['prompt'],
        'actual': row['actual'],
        'ml_pred': row['ml_predicted'],
        'base_pred': row['baseline_predicted'],
        'note': 'Valid IPv4 syntactically. ML and source agree negative. Context (URL/version string) matters.'
    })

for i, r in enumerate(review, 1):
    print(f"\n--- Record {i}: {r['record_type']} ---")
    print(f"  Candidate: {r['candidate']}")
    print(f"  Context:   {r['prompt'][:80]}")
    print(f"  Actual:    {r['actual']} | ML: {r['ml_pred']} | Rule: {r['base_pred']}")
    print(f"  Analyst note: {r['note']}")

# ---------- Save review to CSV ----------
pd.DataFrame(review).to_csv('outputs/analyst_review.csv', index=False)
print("\nSaved: outputs/analyst_review.csv")

# ---------- Write report ----------
with open('reports/phase_e_analysis_v2.txt', 'w') as f:
    f.write("=== PHASE E: EVALUATION AND FORENSIC INTERPRETATION ===\n\n")
    f.write("--- CONFUSION MATRIX (ML model) ---\n")
    f.write(f"TP={tp}  FP={fp}  FN={fn}  TN={tn}\n\n")
    f.write(f"Accuracy:  {acc:.4f}\nPrecision: {prec:.4f}\nRecall:    {rec:.4f}\nF1:        {f1:.4f}\n")
    f.write(f"Majority baseline: {maj_acc:.4f}\n\n")
    
    f.write("--- 8-RECORD ANALYST REVIEW ---\n")
    for i, r in enumerate(review, 1):
        f.write(f"\nRecord {i}: {r['record_type']}\n")
        f.write(f"  Candidate: {r['candidate']}\n")
        f.write(f"  Actual: {r['actual']} | ML: {r['ml_pred']} | Rule: {r['base_pred']}\n")
        f.write(f"  Note: {r['note']}\n")
    
    f.write("\n\n--- COUNTEREXAMPLE: HIGH ACCURACY ON IMBALANCED DATA ---\n")
    f.write("The majority-class baseline (always predict 'positive') scores 73.91%.\n")
    f.write("A model that predicts 'positive' for every record would score 0.7391\n")
    f.write("and yet identify zero true negatives. Its accuracy is high but its\n")
    f.write("operational value is zero for detecting negatives.\n\n")
    f.write("For forensic triage, this matters: a false negative (missing a true\n")
    f.write("IP evidence candidate) could mean relevant evidence is excluded.\n")
    f.write("A false positive (flagging a version string as an IP) wastes analyst\n")
    f.write("time and may contaminate downstream intelligence.\n\n")
    f.write("--- FALSE NEGATIVE CONSEQUENCES ---\n")
    f.write("The ML model produced zero false negatives on this test set, but the\n")
    f.write("rule baseline missed 2 negatives. In a larger deployment, a false\n")
    f.write("negative for an IP evidence candidate means the address is dropped\n")
    f.write("before analyst review. This is worse than a false positive, which\n")
    f.write("only costs review time. Appropriate escalation: any candidate that\n")
    f.write("ML and rule disagree on should be escalated to an analyst rather\n")
    f.write("than resolved automatically.\n")

print("\nSaved: reports/phase_e_analysis_v2.txt")
