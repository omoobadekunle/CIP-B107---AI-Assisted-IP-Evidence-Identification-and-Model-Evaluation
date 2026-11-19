# CIP-B107: AI-Assisted IP Evidence Identification and Model Evaluation

> A forensic examination of a supplied IP dataset using Python, Pandas and Scikit-Learn, completed for the ICDFA CIP-B107 module.

![Platform](https://img.shields.io/badge/Platform-Kali%20Linux-557C94)
![Language](https://img.shields.io/badge/Python-3-3776AB)
![Library](https://img.shields.io/badge/Scikit--Learn-Logistic%20Regression-F7931E)
![Status](https://img.shields.io/badge/Status-Completed-success)

| | |
|---|---|
| **Module** | ICDFA CIP-B107, AI-Assisted IP Evidence Identification and Model Evaluation |
| **Case reference** | CIP-B107-LAB01 |
| **Examiner** | Adekunle Ogunyemi |
| **Environment** | Kali Linux (fully offline, no external API or paid service) |
| **Examination date** | 18 November 2026, 13:50:38 WAT |
| **Time standard** | WAT for examination times, UTC for model timestamps where stated |

---

## Table of Contents

1. [Overview](#overview)
2. [Objectives](#objectives)
3. [Key Results at a Glance](#key-results-at-a-glance)
4. [Tools and Environment](#tools-and-environment)
5. [Evidence Handling and Methodology](#evidence-handling-and-methodology)
6. [Dataset Profile](#dataset-profile)
7. [IP Candidate Extraction and Validation](#ip-candidate-extraction-and-validation)
8. [Offline Machine Learning Classifier](#offline-machine-learning-classifier)
9. [Historical Fine-Tuning Investigation](#historical-fine-tuning-investigation)
10. [Model Evaluation and Analyst Review](#model-evaluation-and-analyst-review)
11. [Key Findings](#key-findings)
12. [Scope and Limitations](#scope-and-limitations)
13. [Recommendations](#recommendations)
14. [Repository Contents](#repository-contents)
15. [Key Learning Outcomes](#key-learning-outcomes)
16. [Author](#author)

---

## Overview

This repository documents my AI-assisted IP evidence identification practical for the ICDFA CIP-B107 module. I preserved a supplied dataset, extracted IP addresses from raw text, trained an offline machine learning classifier, investigated historical fine-tuning logs, evaluated the model, and produced a full forensic report.

My approach was to preserve the source evidence, work only on copies, calculate SHA-256 hashes, extract IP candidates with a regular expression, validate them with Python's `ipaddress` module, and train a Logistic Regression classifier with Scikit-Learn on the supplied JSONL files. I paid particular attention to what the historical training logs actually prove, and to the difference between training accuracy and real-world model reliability.

My conclusions are limited to the records I examined and the test split I used.

## Objectives

- Preserve the supplied evidence and record its integrity.
- Profile the dataset and the JSONL files.
- Extract IP candidates from raw text and distinguish valid IPv4 addresses from misleading strings.
- Build a transparent rule-based baseline and an offline machine learning classifier.
- Investigate the historical fine-tuning logs and interpret what they actually prove.
- Evaluate the model using a confusion matrix, per-class metrics and an 8-record analyst review.
- Produce an executive briefing and final submission package.
- Identify limitations and recommend improvements for future work.

## Key Results at a Glance

| Model | Accuracy | Test records (n) |
|---|---|---|
| Logistic Regression (full holdout) | **97.83%** | 46 |
| Logistic Regression (clean subset, no leaked prompts) | **93.75%** | 16 |
| Rule-based baseline | **95.65%** | 46 |
| Majority-class baseline | **73.91%** | 46 |

- 5 evidence files preserved and verified with SHA-256; all passed the post-examination integrity recheck.
- 229 IP-like strings extracted; 31 rejected as invalid, leaving 198 valid IPv4 addresses.
- 11 of 46 test prompts are exact duplicates of training prompts (train/test leakage), which inflates the headline accuracy.

## Tools and Environment

| Tool | Purpose |
|---|---|
| Kali Linux | Examination environment |
| `sha256sum` | SHA-256 hashing of evidence files |
| Python 3 | Scripting and analysis |
| Pandas | Loading and profiling the dataset and JSONL files |
| `ipaddress` | Validating IPv4 candidates |
| Scikit-Learn | TF-IDF vectorisation and Logistic Regression classifier |
| Matplotlib and Seaborn | Confusion matrix visualisation |
| `zip` | Packaging the final submission |

## Evidence Handling and Methodology

I created a folder structure with separate areas for originals, working copies, scripts, reports, screenshots and outputs. The five core evidence files were copied into originals and left untouched. All analysis was performed on working copies.

| Property | Result |
|---|---|
| Evidence files | `dataset_small.csv`, `dataset_train.jsonl`, `dataset_test.jsonl`, `result.csv`, `fine_tuning_steps.txt` |
| Hash algorithm | SHA-256 |
| Examination time | 18 November 2026, 13:50:38 WAT |
| Integrity recheck | All five files returned OK |
| Additional reference files | `compute_classification_metrics.ipynb`, `preprocess_dataset.py` (also hashed and verified) |

The evidence manifest (`evidence_manifest.csv`) records the filename, SHA-256 hash, file size in bytes, examination time in WAT, examiner name and transfer source for each evidence file.

## Dataset Profile

`dataset_small.csv` has **229 rows** and three columns (`IP`, `class`, `text`) with no missing values.

### Class Distribution

| Class | Count | Approximate share |
|---|---|---|
| A | 177 | 77.3% |
| N | 32 | 14.0% |
| B | 12 | 5.2% |
| C | 4 | 1.7% |
| E | 4 | 1.7% |
| **Total** | **229** | **100%** |

The distribution is heavily uneven, so I relied on per-class metrics and a confusion matrix rather than overall accuracy alone.

### JSONL Profile

| File | Records | Positive | Negative |
|---|---|---|---|
| `dataset_train.jsonl` | 183 | 163 | 20 |
| `dataset_test.jsonl` | 46 | 34 | 12 |

The prompt format is `context;candidate ->`, with a `positive` or `negative` completion. The record counts match the brief's expected 183/46 split.

## IP Candidate Extraction and Validation

A regular expression (four groups of one to three digits separated by dots) produced 229 candidates, but it also matched strings that are not valid IPv4 addresses, such as `1.000.000.000`, `608.80.24.1`, `0.0.0.392`, `257.0.1.38` and software version strings like `6.8.4.2`. A regex alone cannot tell whether a string is a valid IP address, so I passed every candidate through `ipaddress.IPv4Address`, which rejects octets over 255 and leading zeros.

### Three-Way Comparison

| Concept | Count |
|---|---|
| Regex matched a candidate | 229 |
| Syntactically valid IPv4 | 198 |
| Source-labelled positive | 197 |
| Source-labelled negative | 32 |

### Disagreements

| Type | Records | Handling |
|---|---|---|
| Invalid IPv4 but source-labelled positive | 7 | Kept and explained in analyst notes, as the brief states positive labels are not proof of technical validity |
| Invalid IPv4 and source-labelled negative | 24 | Consistent: validator and source agree |
| Valid IPv4 but source-labelled negative | 8 | Includes `0.0.0.0` (x4), `0.5.1.3`, `0.6.7.8` and `0.9.9.3` (x2); syntactically valid but forensically meaningless or a version number in a URL |

All 229 rows were exported to `outputs/candidate_review.csv` with analyst notes.

## Offline Machine Learning Classifier

### Approach

I used Scikit-Learn entirely offline. The input feature was the full prompt string, converted with TF-IDF (word n-grams 1 to 3, token pattern keeping alphanumeric groups and dots together). I trained a Logistic Regression classifier with `class_weight='balanced'`. The vectoriser was fitted on the training records only, so no test vocabulary leaked into the training features.

### Leakage Check

| Check | Result |
|---|---|
| Unique training prompts | 138 |
| Unique test prompts | 25 |
| Exact prompt overlaps | 11 |
| Unique training candidates | 96 |
| Unique test candidates | 23 |
| Candidate-level overlaps | 12 |

About 24% of test prompts (11 of 46) are identical to a training prompt, which any model will trivially get right.

### Results

**Logistic Regression**

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| negative | 1.00 | 0.92 | 0.96 | 12 |
| positive | 0.97 | 1.00 | 0.99 | 34 |
| **Accuracy** | | | **0.9783** | 46 |

**Rule-based baseline** (extract candidate, regex match, then `ipaddress` check)

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| negative | 1.00 | 0.83 | 0.91 | 12 |
| positive | 0.94 | 1.00 | 0.97 | 34 |
| **Accuracy** | | | **0.9565** | 46 |

The rule misses 2 negatives because `0.0.0.0` passes the `ipaddress` check.

### Leakage-Aware Evaluation

| Evaluation | Accuracy | n |
|---|---|---|
| Full test set | 0.9783 | 46 |
| Clean subset (no overlapping prompts) | 0.9375 | 16 |
| Rule baseline (full test set) | 0.9565 | 46 |
| Majority baseline | 0.7391 | 46 |

Removing leaked prompts drops accuracy by about 4 percentage points, so the headline number is optimistic.

## Historical Fine-Tuning Investigation

I examined `fine_tuning_steps.txt` (49 rows, a truncated view) and `result.csv` (733 rows).

| Property | Value |
|---|---|
| Total steps | 733 |
| Total elapsed tokens | 34,661 |
| Total elapsed examples | 733 |

- Training accuracy reaches 1.0 by step 5 and stays there, which is textbook overfitting (memorisation).
- Validation loss was recorded 92 times in 733 steps; 82 checks showed sequence accuracy of 1.0 and 10 were below 1.0.

**Final classification metrics (step 733)**

| Metric | Value |
|---|---|
| Accuracy | 0.978261 |
| Precision | 0.971429 |
| Recall | 1.000000 |
| F1 | 0.985507 |
| AUROC | 0.997549 |
| AUPRC | 0.999147 |

The historical accuracy matches my fresh model because both are evaluated on the same test set, which contains the same 11 leaked prompts. This is a **historical result supplied by the module, not a model I trained**.

**What I can conclude:** the model memorised the training set; the final metrics exist in `result.csv`; they were measured on a test set with training overlaps.

**What I cannot conclude:** real-world reliability, the training/validation split composition, or whether the result would reproduce at a different step count.

> **Security note:** The supplied legacy scripts use obsolete endpoints and contain embedded credentials. I did not run them and did not copy any token into this report.

## Model Evaluation and Analyst Review

### Confusion Matrix (ML Model)

| | Predicted negative | Predicted positive |
|---|---|---|
| **Actual negative** | 11 (TN) | 1 (FP) |
| **Actual positive** | 0 (FN) | 34 (TP) |

### Eight-Record Analyst Review

| # | Candidate | Actual | ML | Rule | Outcome |
|---|---|---|---|---|---|
| 1 | `2.20.31.4` | positive | positive | positive | TP, both agree |
| 2 | `13.3.1.17` | positive | positive | positive | TP, both agree |
| 3 | `608.80.24.1` | negative | negative | negative | TN, invalid octet |
| 4 | `257.0.1.38` | negative | negative | negative | TN, invalid octet |
| 5 | `0.0.0.0` | negative | negative | positive | TN, ML beat rule |
| 6 | `0.0.0.0` | negative | negative | positive | TN, ML beat rule |
| 7 | `256.0.1.26` | negative | positive | negative | FP, rule beat ML |
| 8 | additional negative | negative | negative | negative | TN, both correct |

- The ML model beat the rule on the two `0.0.0.0` cases and failed on `256.0.1.26`, since it has no notion of octet range. The two approaches have complementary failure modes.
- **Imbalanced-data counterexample:** the majority baseline scores 73.91%, so most of the headline accuracy comes from predicting the common class.
- **False negatives** (a real IP dropped before analyst review) are more serious than false positives. The ML model produced 0 false negatives; the rule baseline produced 2.
- **Escalation:** where ML and rule disagree (records 5, 6 and 7), the record should go to an analyst, since automated resolution would have been wrong in all three.

## Key Findings

- Evidence integrity was preserved and verified for all five files.
- The JSONL split matches the brief: 183 training and 46 testing records.
- The ML classifier is not dramatically better than the rule baseline; well-designed rules get most of the way there.
- Headline accuracy is inflated by train/test leakage (11 of 46 prompts).
- The ML blind spot is IP validity; the rule blind spot is forensic meaning.
- Historical logs show memorisation, not generalisation.

**Alternative explanations considered**

- The 97.83% accuracy may be partly due to leaked examples; the 93.75% clean-subset result supports this.
- The `0.0.0.0` wins for ML may be a lucky pattern match; with only two examples this cannot be separated from a learned context rule.
- The `256.0.1.26` failure may be a fundamental limitation of the approach rather than a fixable training gap.

## Scope and Limitations

- Findings rest on a single fixed 183/46 split; a different split could produce different results.
- 11 of 46 test prompts duplicate training prompts, so 97.83% should be read alongside 93.75%.
- The classifier used only prompt text as a feature.
- The historical logs do not identify the fine-tuning dataset or how its splits were built.
- The model treats IPs as text patterns and does not validate them. In a production pipeline, strict validation would run first and flag `256.0.1.26` before it reached the model.

## Recommendations

- Deploy the model as **analyst assistance only**, not automated decision-making.
- Route ML/rule disagreements to an analyst.
- Never classify a recovered address as malicious without corroborating evidence.
- Build a clean holdout by removing the 11 overlapping prompts before training.
- Expand the negative class with context-aware examples (URLs, version strings, wildcard addresses).
- Add a hard `ipaddress` validation step in front of the model.
- Review historical fine-tuning scripts for credential removal and API updates before any use.

## Repository Contents

```text
CIP-B107-LAB01/
├── README.txt / README.md               # This file
├── Report/                              # Full forensic report (DOCX)
├── evidence_manifest.csv                # SHA-256 manifest of evidence files
├── outputs/
│   ├── candidate_review.csv             # 229 candidates with analyst notes
│   ├── predictions.csv                  # Classifier predictions
│   ├── analyst_review.csv               # Eight-record analyst review
│   ├── phase_c_classification_report.txt
│   ├── phase_c_confusion_matrix_v2.png
│   └── phase_d_loss_plot.png
└── reports/
    └── phase_d_historical_analysis.txt
```

> Adjust this tree to match the final layout of your repository.

## Key Learning Outcomes

- Preserving evidence and recording integrity with hashes and a chain-of-custody manifest.
- Profiling datasets with Pandas and JSONL files with Python's `json` module.
- Extracting candidate IP addresses with regular expressions and rejecting false positives with `ipaddress`.
- Training an offline classifier with Scikit-Learn and TF-IDF vectorisation.
- Evaluating with per-class metrics, a confusion matrix and a majority baseline.
- Distinguishing training accuracy from real-world reliability.
- Detecting and disclosing train/test leakage and its effect on measured accuracy.
- Tracing misclassifications to specific records and explaining them in forensic terms.

## Author

**Adekunle Ogunyemi**  
Digital Forensics Examiner | ICDFA CIP-B107

*This work was completed for educational purposes as part of the ICDFA programme.*